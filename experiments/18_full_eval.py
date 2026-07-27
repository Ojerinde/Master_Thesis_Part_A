"""
Single-run comprehensive adversarial evaluation at the common recall=0.95
operating point, so every adversarial number in the paper comes from ONE run and
ONE operating point (eliminates the mixed-run inconsistency).

For all 13 models, on a fixed stratified test subsample, at each model's
recall=0.95 threshold, it computes clean + adversarial recall/F1/ASR under:
  white-box FGSM, PGD           (DL only)
  transfer FGSM/PGD, single(CNN-1D) and multi-surrogate(CNN-1D+LSTM+TCN)  (classical)
  DLSA, SNA, TPA                (all)
Decision-based boundary ASR is merged from blackbox_boundary_all.csv.

Output: results/tables/adversarial_full_oppoint.csv  (master table).
Run:    PYTHONPATH=. python experiments/18_full_eval.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from sklearn.preprocessing import MinMaxScaler                # noqa: E402
from sklearn.model_selection import train_test_split            # noqa: E402
from sklearn.metrics import recall_score, f1_score              # noqa: E402
from data.loader import load_track_splits                       # noqa: E402
from config.paths import CLASSICAL_MODELS, DL_MODELS, TABLES_DIR  # noqa: E402
from config.model_configs import get_config                     # noqa: E402
from models.deep_learning import (                               # noqa: E402
    CNN1DModel, LSTMModel, BiLSTMModel, CNNLSTMModel, TransformerModel, TCNModel)
from attacks.fgsm import FGSMAttack                              # noqa: E402
from attacks.pgd import PGDAttack                                # noqa: E402
from attacks.gnss_attacks import (                               # noqa: E402
    DataLocationShiftAttack, SimilarityNoiseAttack, TemporalPatternAttack)
from utils.gnss_constraints import GNSSConstraintEnforcer         # noqa: E402

EPS = [0.05, 0.10, 0.20]
TARGET_RECALL = 0.95
N_EVAL = 4000
CLASSICAL = {'RandomForest': 'RandomForest_default', 'XGBoost': 'XGBoost_default',
             'LightGBM': 'LightGBM_default', 'GradientBoosting': 'GradientBoosting',
             'KNN': 'KNN', 'MLP': 'MLP', 'DecisionTree': 'DecisionTree'}
DL = {'CNN-1D': (CNN1DModel, 'cnn_1d'), 'LSTM': (LSTMModel, 'lstm'),
      'BiLSTM': (BiLSTMModel, 'bilstm'), 'CNN-LSTM': (CNNLSTMModel, 'cnn_lstm'),
      'Transformer': (TransformerModel, 'transformer'), 'TCN': (TCNModel, 'tcn')}


class Ens(nn.Module):
    def __init__(self, mods):
        super().__init__(); self.mods = nn.ModuleList(mods)

    def forward(self, x):
        return torch.stack([m(x) for m in self.mods], 0).mean(0)


class W:
    def __init__(self, m): self.model = m


def flat(p):
    p = np.asarray(p)
    if p.ndim == 2:
        p = p[:, 1] if p.shape[1] >= 2 else p[:, 0]
    return p.ravel().astype(np.float64)


def tau_for(yv, pv):
    best, fb, br = None, 0.5, -1
    for t in np.linspace(0.01, 0.99, 197):
        r = recall_score(yv, (pv >= t).astype(int), zero_division=0)
        if r >= TARGET_RECALL:
            best = t
        if r > br:
            br, fb = r, t
    return float(best) if best is not None else float(fb)


def load():
    # Leakage-free block-temporal split (shared load_track_splits); the 9 FGI
    # observables used directly (no engineering). Scaler fit on the train block.
    (Xtr_e, Xv_e, Xte_e, _ytr, yv, yte, fn, sc) = load_track_splits(verbose=False)
    return (Xtr_e.astype(np.float64), Xv_e.astype(np.float64), Xte_e.astype(np.float64),
            sc, fn, yv, yte)


def metrics(y, p, tau):
    pred = (p >= tau).astype(int)
    return recall_score(y, pred, zero_division=0), f1_score(y, pred, zero_division=0)


def asr_of(p_clean, p_adv, y, tau):
    pc = (p_clean >= tau).astype(int); pa = (p_adv >= tau).astype(int)
    cs = (y == 1) & (pc == 1)
    return float(np.mean(pa[cs] == 0)) if cs.sum() else np.nan


def couple_scaled(Xa, sc, enf_phys):
    """Re-apply the C/N0~I/Q-power coupling to a SCALED adversarial batch: inverse-
    transform to physical units, clip C/N0 to the residual band of the (perturbed)
    correlator power, re-transform. enf_phys must be fit_coupling()'d on unscaled
    training data. This is the realizability gate the box-clip alone does not
    provide (attacks only call clip_to_gnss_bounds internally)."""
    Xa_phys = sc.inverse_transform(Xa).astype(np.float64)
    Xa_phys = enf_phys.enforce_coupling_phys(Xa_phys)
    return sc.transform(Xa_phys).astype(np.float32)


def main():
    Xtr_e, Xv_e, Xte_e, sc, fn, yv, yte = load()
    Xtr_s = sc.transform(Xtr_e).astype(np.float32)
    enf_dl = GNSSConstraintEnforcer(fn); enf_dl.fit(Xtr_s, fn)
    enf_clf = GNSSConstraintEnforcer(fn); enf_clf.fit(Xtr_e.astype(np.float32), fn)
    enf_clf.fit_coupling(Xtr_e, fn)

    rng = np.random.default_rng(42)
    idx = np.concatenate([rng.choice(np.where(yte == c)[0],
                          int(round(N_EVAL * (yte == c).mean())), replace=False)
                          for c in (0, 1)])
    ys = yte[idx]
    Xs_e = Xte_e[idx].astype(np.float64)
    Xs_s = sc.transform(Xs_e).astype(np.float32)

    # DL surrogates for transfer.
    dl_mods = {}
    for name, (cls, cfg) in DL.items():
        c = get_config(cfg); c['input_dim'] = Xs_s.shape[1]
        m = cls(input_dim=Xs_s.shape[1], config=c); m.build_model()
        m.model.load_state_dict(torch.load(str(DL_MODELS / f'{cfg}.pt'), map_location='cpu'))
        m.model.eval(); m.is_trained = True; dl_mods[name] = m

    rows = []

    def dom_attacks(space, y, enf, eps):
        out = {}
        d = DataLocationShiftAttack(shift_scale=eps, feature_names=fn, gnss_enforcer=enf)
        out['DLSA'] = d.generate(space, y)
        s = SimilarityNoiseAttack(epsilon=eps, gnss_enforcer=enf).fit(space)
        out['SNA'] = s.generate(space, y)
        t = TemporalPatternAttack(doppler_amp=eps, cn0_amp=eps * 0.6,
                                  feature_names=fn, gnss_enforcer=enf)
        out['TPA'] = t.generate(space, y)
        return out

    # ---- DL models ----
    for name, m in dl_mods.items():
        pv = flat(m.predict_proba(sc.transform(Xv_e).astype(np.float32)))
        tau = tau_for(yv, pv)
        pc = flat(m.predict_proba(Xs_s)); rc, fc = metrics(ys, pc, tau)
        rows.append(dict(model=name, family='deep', attack='clean', eps=0,
                         recall=round(rc, 4), f1=round(fc, 4), asr=np.nan, tau=round(tau, 4)))
        for eps in EPS:
            fg = FGSMAttack(m, epsilon=eps, gnss_enforcer=enf_dl).generate(Xs_s, ys)
            pg = PGDAttack(m, epsilon=eps, num_iter=40, random_start=True,
                           gnss_enforcer=enf_dl).generate(Xs_s, ys)
            dom = dom_attacks(Xs_s, ys, enf_dl, eps)
            for an, Xa in [('FGSM', fg), ('PGD', pg), ('DLSA', dom['DLSA']),
                           ('SNA', dom['SNA']), ('TPA', dom['TPA'])]:
                Xa = couple_scaled(Xa, sc, enf_clf)
                pa = flat(m.predict_proba(Xa.astype(np.float32)))
                r, f = metrics(ys, pa, tau)
                rows.append(dict(model=name, family='deep', attack=an, eps=eps,
                                 recall=round(r, 4), f1=round(f, 4),
                                 asr=round(asr_of(pc, pa, ys, tau), 4), tau=round(tau, 4)))

    # ---- Classical models: transfer + domain-specific ----
    ens = W(Ens([dl_mods['CNN-1D'].model, dl_mods['LSTM'].model, dl_mods['TCN'].model]))
    single = W(dl_mods['CNN-1D'].model)
    # Precompute transfer adversarials in scaled space, inverse-transform to eng.
    transfer_eng = {}
    for eps in EPS:
        for tag, wrap in [('single', single), ('multi', ens)]:
            adv = PGDAttack(wrap, epsilon=eps, num_iter=40, random_start=True,
                            gnss_enforcer=enf_dl).generate(Xs_s, ys)
            adv_phys = sc.inverse_transform(adv).astype(np.float64)
            transfer_eng[(tag, eps)] = enf_clf.enforce_coupling_phys(adv_phys)

    for name, stem in CLASSICAL.items():
        mdl = joblib.load(CLASSICAL_MODELS / f'{stem}.joblib')
        pv = flat(mdl.predict_proba(Xv_e)); tau = tau_for(yv, pv)
        pc = flat(mdl.predict_proba(Xs_e)); rc, fc = metrics(ys, pc, tau)
        rows.append(dict(model=name, family='classical', attack='clean', eps=0,
                         recall=round(rc, 4), f1=round(fc, 4), asr=np.nan, tau=round(tau, 4)))
        for eps in EPS:
            # Domain attacks generated in the SAME min-max [0,1] space as the deep
            # models (so the budget is a min-max L-inf distance), then inverse-
            # transformed to physical for the classical Pipeline, which re-applies
            # the identical MinMax internally (a round trip). This removes the old
            # unscaled-space classical budget that was not comparable across models.
            dom = dom_attacks(Xs_s, ys, enf_dl, eps)
            dom = {k: sc.inverse_transform(v).astype(np.float64) for k, v in dom.items()}
            items = [('PGD-Transfer', transfer_eng[('single', eps)]),
                     ('PGD-Transfer-Multi', transfer_eng[('multi', eps)]),
                     ('DLSA', dom['DLSA']), ('SNA', dom['SNA']), ('TPA', dom['TPA'])]
            for an, Xa in items:
                Xa = enf_clf.enforce_coupling_phys(Xa.astype(np.float64))
                pa = flat(mdl.predict_proba(Xa.astype(np.float64)))
                r, f = metrics(ys, pa, tau)
                rows.append(dict(model=name, family='classical', attack=an, eps=eps,
                                 recall=round(r, 4), f1=round(f, 4),
                                 asr=round(asr_of(pc, pa, ys, tau), 4), tau=round(tau, 4)))

    df = pd.DataFrame(rows)
    out = TABLES_DIR / 'adversarial_full_oppoint.csv'
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} rows, N_eval={len(ys)})")
    # quick worst-case recall per model (min over all attacks incl. decision-based)
    bb = pd.read_csv(TABLES_DIR / 'blackbox_boundary_all.csv')
    print("\n-- adv recall @eps=0.20 by attack (min = worst) --")
    piv = df[df.eps == 0.20].pivot_table(index='model', columns='attack', values='recall')
    print(piv.round(3).to_string())


if __name__ == '__main__':
    main()
