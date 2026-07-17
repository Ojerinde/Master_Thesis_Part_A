"""
Diagnostic defense baseline (Paper 1) - empirical adversarial training.
=======================================================================
Purpose: answer the reviewer question "does a standard defense fix this?" WITHOUT
claiming a certificate. This is the EMPIRICAL baseline (adversarial training on the
deep detectors), NOT the certified randomized-smoothing / GNSS-Shield defense, which
is Paper 2 and is out of scope here.

Scheme: PGD adversarial training (Madry et al. 2018) done to current best practice.
For each deep model we train (1) an undefended baseline by standard training and
(2) a defended model by min-max PGD-AT: adversarial examples are generated ON THE FLY
each mini-batch against the CURRENT weights (inner maximisation) and the network is
trained to classify them correctly (outer minimisation). Crucially, the checkpoint is
selected by EARLY STOPPING on a ROBUST validation metric (spoof recall under PGD on the
validation block at the recall=0.95 operating point), which is the established fix for
robust overfitting (Rice et al. 2020) - selecting on clean loss is exactly the mistake
that makes AT look ineffective. We then evaluate BOTH variants at the common recall=0.95
operating point under clean, FGSM and PGD. FGSM/PGD at eval are regenerated against each
variant, so the attack on the defended model is ADAPTIVE (white-box against the defended
weights), the correct test of whether the defense holds (Athalye 2018; Carlini 2019).

Design choices (reviewer-facing, per the AT literature):
  - inner PGD: random start, L-inf budget AUG_EPS, AT_PGD_STEPS steps (eval uses 40).
  - outer: Adam, weight decay 5e-4 (Rice 2020 setting).
  - selection: best ROBUST-val checkpoint, patience AT_PATIENCE (guards robust overfit).
  - every adversarial example (train and eval) is projected to the physically realizable
    set by the same enforcer used elsewhere, so AT defends the realizable threat.

Expected, and coherent with the paper's thesis: adversarial training recovers some
robustness against the perturbation family it trains on, at a small clean-accuracy
cost, but the detector remains vulnerable to the adaptive attack. That motivates a
principled/certified defense (future work) without over-claiming.

Classical models are excluded: adversarial training is a gradient-based procedure with
no standard analogue for tree/kNN detectors; their defense is left to future work.

Output: results/tables/defense_baseline.csv
Run (GPU strongly recommended, e.g. Kaggle):
    PYTHONPATH=. python experiments/24_defense.py
    PYTHONPATH=. python experiments/24_defense.py --smoke     # fast CPU sanity
"""
from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from sklearn.metrics import recall_score, f1_score              # noqa: E402
from data.loader import load_track_splits                       # noqa: E402
from config.paths import TABLES_DIR                             # noqa: E402
from config.model_configs import get_config                     # noqa: E402
from models.deep_learning import (                               # noqa: E402
    CNN1DModel, LSTMModel, BiLSTMModel, CNNLSTMModel, TransformerModel, TCNModel)
from attacks.fgsm import FGSMAttack                              # noqa: E402
from attacks.pgd import PGDAttack                                # noqa: E402
from attacks.gnss_attacks import (                               # noqa: E402
    DataLocationShiftAttack, SimilarityNoiseAttack, TemporalPatternAttack)
from utils.gnss_constraints import GNSSConstraintEnforcer         # noqa: E402

DL = {'CNN-1D': (CNN1DModel, 'cnn_1d'), 'LSTM': (LSTMModel, 'lstm'),
      'BiLSTM': (BiLSTMModel, 'bilstm'), 'CNN-LSTM': (CNNLSTMModel, 'cnn_lstm'),
      'Transformer': (TransformerModel, 'transformer'), 'TCN': (TCNModel, 'tcn')}
EPS = [0.05, 0.10, 0.20]
TARGET_RECALL = 0.95
AUG_EPS = 0.10          # inner-maximization L-inf budget for PGD adversarial training
AT_PGD_STEPS = 10       # PGD steps for the inner max during training (eval uses 40)
AT_MAX_EPOCHS = 30      # cap; early stopping on ROBUST val usually stops sooner
AT_PATIENCE = 5         # robust-val patience (Rice 2020: guards against robust overfitting)
AT_BATCH = 256
N_EVAL = 4000


def set_determinism(seed=42):
    import torch, random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flat(p):
    p = np.asarray(p)
    return (p[:, 1] if p.ndim == 2 and p.shape[1] >= 2 else p.ravel()).astype(np.float64)


def tau_for(yv, pv):
    best, fb, br = None, 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 197):
        r = recall_score(yv, (pv >= t).astype(int), zero_division=0)
        if r >= TARGET_RECALL:
            best = t
        if r > br:
            br, fb = r, t
    return float(best) if best is not None else float(fb)


def metrics(y, p, tau):
    pred = (p >= tau).astype(int)
    return recall_score(y, pred, zero_division=0), f1_score(y, pred, zero_division=0)


def asr_of(p_clean, p_adv, y, tau):
    pc = (p_clean >= tau).astype(int); pa = (p_adv >= tau).astype(int)
    cs = (y == 1) & (pc == 1)
    return float(np.mean(pa[cs] == 0)) if cs.sum() else np.nan


def build_train(cls, cfg_name, Xtr, ytr, Xv, yv, epochs=None, batch_size=None):
    c = get_config(cfg_name); c['input_dim'] = Xtr.shape[1]
    if epochs is not None:
        c['epochs'] = epochs
    if batch_size is not None:
        c['batch_size'] = batch_size
    m = cls(input_dim=Xtr.shape[1], config=c); m.build_model()
    m.train(Xtr, ytr, X_val=Xv, y_val=yv)
    return m


def couple_scaled(Xa, sc, enf_phys):
    Xp = sc.inverse_transform(Xa).astype(np.float64)
    Xp = enf_phys.enforce_coupling_phys(Xp)
    return sc.transform(Xp).astype(np.float32)


def dom_attacks(space, y, enf, fn, eps):
    """The three GNSS domain attacks in scaled space (same recipe as 18_full_eval).
    Attack families named by An et al. (2025); here in the observable domain."""
    out = {}
    out['DLSA'] = DataLocationShiftAttack(
        shift_scale=eps, feature_names=fn, gnss_enforcer=enf).generate(space, y)
    out['SNA'] = SimilarityNoiseAttack(epsilon=eps, gnss_enforcer=enf).fit(space).generate(space, y)
    out['TPA'] = TemporalPatternAttack(
        doppler_amp=eps, cn0_amp=eps * 0.6, feature_names=fn,
        gnss_enforcer=enf).generate(space, y)
    return out


def robust_val_recall(m, Xv_s, yv, enf, enf_phys, sc, eps, steps, tau):
    """Spoof recall under PGD(eps) on the validation block. This is the model-selection
    metric that guards against robust overfitting (Rice et al. 2020); selecting on it
    rather than on clean loss is what makes AT effective."""
    adv = PGDAttack(m, epsilon=eps, num_iter=steps, random_start=True,
                    gnss_enforcer=enf).generate(Xv_s, yv)
    adv = couple_scaled(adv, sc, enf_phys)
    pa = flat(m.predict_proba(adv.astype(np.float32)))
    return recall_score(yv, (pa >= tau).astype(int), zero_division=0)


def adv_train_model(cls, cfg_name, Xtr_s, ytr, Xv_s, yv, enf, enf_phys, sc,
                    eps, pgd_steps, max_epochs, patience, batch=AT_BATCH):
    """PGD adversarial training (Madry 2018) with robust-validation early stopping
    (Rice 2020). Adversarial examples are regenerated on the fly each mini-batch
    against the current weights; the checkpoint with the best robust-val spoof recall
    is restored. Uses the model's own nn.Module so no model file is modified."""
    import torch
    import torch.nn as nn
    import copy
    from torch.utils.data import TensorDataset, DataLoader

    c = get_config(cfg_name); c['input_dim'] = Xtr_s.shape[1]
    m = cls(input_dim=Xtr_s.shape[1], config=c); m.build_model()
    net, dev = m.model, m.device
    opt = torch.optim.Adam(net.parameters(),
                           lr=c.get('learning_rate', 1e-3), weight_decay=5e-4)
    crit = nn.BCEWithLogitsLoss()
    atk = PGDAttack(m, epsilon=eps, num_iter=pgd_steps, random_start=True,
                    gnss_enforcer=enf)
    ds = TensorDataset(torch.tensor(Xtr_s, dtype=torch.float32),
                       torch.tensor(ytr, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)

    best_rob, best_state, ctr = -1.0, None, 0
    for epoch in range(max_epochs):
        for Xb, yb in dl:
            # inner maximisation: PGD adversarial batch vs current weights, projected
            # to the physically realizable set (same enforcer used at eval)
            Xadv = atk.generate(Xb.numpy(), yb.numpy().astype(int))
            Xadv = couple_scaled(Xadv, sc, enf_phys)
            # outer minimisation: train to classify the adversarial batch correctly
            net.train()
            xb = torch.tensor(Xadv, dtype=torch.float32).to(dev)
            opt.zero_grad()
            loss = crit(net(xb), yb.to(dev))
            loss.backward(); opt.step()
        # robust-val model selection at the recall-0.95 operating point
        net.eval()
        tau = tau_for(yv, flat(m.predict_proba(Xv_s)))
        rob = robust_val_recall(m, Xv_s, yv, enf, enf_phys, sc, eps, pgd_steps, tau)
        if rob > best_rob:
            best_rob, best_state, ctr = rob, copy.deepcopy(net.state_dict()), 0
        else:
            ctr += 1
        print(f"    AT epoch {epoch + 1}: robust-val recall={rob:.3f} "
              f"(best={best_rob:.3f}, patience {ctr}/{patience})", flush=True)
        if ctr >= patience:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    m.is_trained = True
    return m


def boundary_attack(proba_eng, Xs, anchors, tau, fmin, span, n_line=18, n_clip=18):
    """Vectorized decision-based L-inf boundary attack in [0,1] min-max space, copied
    verbatim from 13_blackbox_attacks.py so the defended-model numbers use the SAME
    gradient-free attack (and the SAME space) as the main results. Returns the minimal
    L-inf that flips each spoof sample to authentic."""
    def is_auth(Xn):
        return proba_eng(Xn * span + fmin) < tau
    N = Xs.shape[0]
    best_linf = np.full(N, np.inf); best_adv = Xs.copy()
    for a in anchors:
        direction = a[None, :] - Xs
        t_lo, t_hi = np.zeros(N), np.ones(N)
        for _ in range(n_line):
            t_mid = 0.5 * (t_lo + t_hi)
            auth = is_auth(Xs + t_mid[:, None] * direction)
            t_hi = np.where(auth, t_mid, t_hi); t_lo = np.where(auth, t_lo, t_mid)
        delta = t_hi[:, None] * direction
        c_lo = np.zeros(N); c_hi = np.max(np.abs(delta), axis=1) + 1e-9
        for _ in range(n_clip):
            c_mid = 0.5 * (c_lo + c_hi)
            dc = np.clip(delta, -c_mid[:, None], c_mid[:, None])
            auth = is_auth(np.clip(Xs + dc, 0.0, 1.0))
            c_hi = np.where(auth, c_mid, c_hi); c_lo = np.where(auth, c_lo, c_mid)
        dc = np.clip(delta, -c_hi[:, None], c_hi[:, None])
        adv = np.clip(Xs + dc, 0.0, 1.0)
        better = c_hi < best_linf
        best_linf = np.where(better, c_hi, best_linf)
        best_adv = np.where(better[:, None], adv, best_adv)
    return best_adv, best_linf


def decision_based_rows(m, name, tag, Xv_e, yv, Xte_e_sub, y_sub, fmin, span, sc,
                        n_anchors=8):
    """Decision-based (gradient-free) evaluation of one model in the [0,1] space, so a
    defended model that resists PGD is still tested by the attack that defeats gradient
    masking. This is what tells us whether adversarial training actually closes the gap
    or only the gradient-attack surface."""
    def proba_eng(Xe):
        return flat(m.predict_proba(sc.transform(Xe.astype(np.float64)).astype(np.float32)))

    def to_norm(X):
        return np.clip((X - fmin) / span, 0.0, 1.0)

    tau = tau_for(yv, proba_eng(Xv_e))
    p = proba_eng(Xte_e_sub)
    cs = np.where((y_sub == 1) & (p >= tau))[0]
    if len(cs) == 0:
        return []
    Xs = to_norm(Xte_e_sub[cs])
    ap = np.where((y_sub == 0) & (p < tau))[0]
    if len(ap) == 0:
        # fallback: if no genuine point sits below tau (e.g. a high-FAR operating
        # point), use the most authentic-looking genuine samples (lowest proba) so
        # the boundary attack always has valid targets rather than returning NaN.
        genuine = np.where(y_sub == 0)[0]
        if len(genuine) == 0:
            return []
        ap = genuine[np.argsort(p[genuine])[:max(n_anchors, 1)]]
    rng = np.random.default_rng(42)
    a_idx = rng.choice(ap, size=min(n_anchors, len(ap)), replace=False)
    anchors = to_norm(Xte_e_sub[a_idx])
    adv, linf = boundary_attack(proba_eng, Xs, anchors, tau, fmin, span)
    flipped = proba_eng(adv * span + fmin) < tau
    rows = []
    for eps in [0.05, 0.10, 0.20, float('inf')]:
        within = linf <= eps + 1e-9
        rows.append(dict(model=name, defense=tag, attack='Decision-based', eps=eps,
                         recall=np.nan, f1=np.nan,
                         asr=round(float(np.mean(flipped & within)), 4),
                         tau=round(tau, 4),
                         median_min_linf=round(float(np.median(linf)), 4)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true',
                    help='fast sanity: 2 models, small subsample, few epochs (CPU ok)')
    ap.add_argument('--models', default=None,
                    help='comma-separated subset of the 6 DL models to run this '
                         'session (e.g. "CNN-1D,LSTM"), to chunk across multiple '
                         'Kaggle 12h sessions the same way 23_generalization.py '
                         'does. Default: all not-yet-done models.')
    ap.add_argument('--batch-size', type=int, default=None,
                    help='override the DL batch size (config default 32 under-uses '
                         'a GPU); PGD-AT is ~10x more expensive per epoch than plain '
                         'training (AT_PGD_STEPS inner steps per batch), so this '
                         'matters even more here than in 23_generalization.py')
    args = ap.parse_args()
    set_determinism(42)

    (Xtr_e, Xv_e, Xte_e, ytr, yv, yte, fn, sc) = load_track_splits(verbose=True)
    Xtr_e = Xtr_e.astype(np.float64)
    Xtr_s = sc.transform(Xtr_e).astype(np.float32)
    Xv_s = sc.transform(Xv_e).astype(np.float32)

    enf = GNSSConstraintEnforcer(fn); enf.fit(Xtr_s, fn)
    enf_phys = GNSSConstraintEnforcer(fn); enf_phys.fit(Xtr_e.astype(np.float32), fn)
    enf_phys.fit_coupling(Xtr_e, fn)

    # stratified test subsample for the attack eval (same recipe as 18/20)
    rng = np.random.default_rng(42)
    idx = np.concatenate([rng.choice(np.where(yte == c)[0],
                          int(round(N_EVAL * (yte == c).mean())), replace=False)
                          for c in (0, 1)])
    ys = yte[idx]
    Xs_s = sc.transform(Xte_e[idx].astype(np.float64)).astype(np.float32)
    Xse_e = Xte_e[idx].astype(np.float64)               # engineered subsample (decision-based)
    fmin = Xtr_e.min(axis=0)                            # [0,1] min-max for the boundary attack
    span = np.maximum(Xtr_e.max(axis=0) - fmin, 1e-12)

    models = DL
    epochs = None
    if args.smoke:
        models = {k: DL[k] for k in ('CNN-1D', 'LSTM')}
        epochs = 3
        tr = rng.choice(len(ytr), min(8000, len(ytr)), replace=False)
        Xtr_s, ytr, Xtr_e = Xtr_s[tr], ytr[tr], Xtr_e[tr]
    elif args.models:
        wanted = [m.strip() for m in args.models.split(',')]
        unknown = [m for m in wanted if m not in DL]
        if unknown:
            print(f"Unknown model(s) in --models: {unknown}. Known: {list(DL)}")
            sys.exit(2)
        models = {k: DL[k] for k in wanted}

    bs = args.batch_size

    # Checkpointing: PGD-AT is ~10x more expensive per epoch than plain training
    # (AT_PGD_STEPS inner steps per batch), so all 6 models can easily exceed a
    # Kaggle 12h session (a timeout kill preserves NOTHING -- see
    # 23_generalization.py for the same lesson learned). Write after EVERY model
    # and skip models already present, so --models chunks safely across sessions.
    outp = TABLES_DIR / ('defense_baseline_smoke.csv' if args.smoke else 'defense_baseline.csv')
    rows = []
    done_models = set()
    if outp.exists() and not args.smoke:
        prev = pd.read_csv(outp)
        rows = prev.to_dict('records')
        done_models = set(prev['model'].unique())
        print(f"Resuming: {outp.name} exists with {len(prev)} rows "
              f"({len(done_models)} model(s) already done: {sorted(done_models)}) "
              f"-- they will be skipped.")

    def checkpoint():
        pd.DataFrame(rows).to_csv(outp, index=False)

    if args.smoke:
        at_steps, at_maxep, at_pat = 3, 3, 2
    else:
        at_steps, at_maxep, at_pat = AT_PGD_STEPS, AT_MAX_EPOCHS, AT_PATIENCE
    for name, (cls, cfg) in models.items():
        if name in done_models:
            print(f"\n=== {name} ===  SKIP (already in {outp.name})", flush=True)
            continue
        print(f"\n=== {name} ===", flush=True)
        # undefended baseline (standard training)
        base = build_train(cls, cfg, Xtr_s, ytr, Xv_s, yv, epochs, batch_size=bs)
        # defended: proper PGD adversarial training with robust-val early stopping
        deff = adv_train_model(cls, cfg, Xtr_s, ytr, Xv_s, yv, enf, enf_phys, sc,
                               eps=AUG_EPS, pgd_steps=at_steps,
                               max_epochs=at_maxep, patience=at_pat,
                               batch=bs or AT_BATCH)

        for tag, m in [('none', base), ('adv_train', deff)]:
            tau = tau_for(yv, flat(m.predict_proba(Xv_s)))
            pc = flat(m.predict_proba(Xs_s)); rc, fc = metrics(ys, pc, tau)
            far = float(((pc >= tau) & (ys == 0)).sum() / max((ys == 0).sum(), 1))
            rows.append(dict(model=name, defense=tag, attack='clean', eps=0.0,
                             recall=round(rc, 4), f1=round(fc, 4), asr=np.nan,
                             tau=round(tau, 4), far=round(far, 4)))
            # full white-box + GNSS domain suite (same as 18_full_eval, DL branch),
            # regenerated against THIS variant so the attack on the defended model is
            # adaptive (white-box against the defended weights)
            for eps in EPS:
                fg = FGSMAttack(m, epsilon=eps, gnss_enforcer=enf).generate(Xs_s, ys)
                pg = PGDAttack(m, epsilon=eps, num_iter=40, random_start=True,
                               gnss_enforcer=enf).generate(Xs_s, ys)
                dm = dom_attacks(Xs_s, ys, enf, fn, eps)
                for an, Xa in [('FGSM', fg), ('PGD', pg), ('DLSA', dm['DLSA']),
                               ('SNA', dm['SNA']), ('TPA', dm['TPA'])]:
                    Xa = couple_scaled(Xa, sc, enf_phys)
                    pa = flat(m.predict_proba(Xa))
                    r, f = metrics(ys, pa, tau)
                    rows.append(dict(model=name, defense=tag, attack=an, eps=eps,
                                     recall=round(r, 4), f1=round(f, 4),
                                     asr=round(asr_of(pc, pa, ys, tau), 4),
                                     tau=round(tau, 4)))
            # decision-based (gradient-free) in [0,1] space - the key test: does AT
            # survive the attack that defeats gradient masking, or only the gradients?
            rows.extend(decision_based_rows(m, name, tag, Xv_e, yv, Xse_e, ys,
                                            fmin, span, sc))
            db = [x for x in rows if x['model'] == name and x['defense'] == tag
                  and x['attack'] == 'Decision-based' and x['eps'] == 0.10]
            pgd = [x for x in rows if x['model'] == name and x['defense'] == tag
                   and x['attack'] == 'PGD' and x['eps'] == 0.10]
            print(f"  {name}/{tag}: clean R={rc:.3f}  "
                  f"PGD@0.1 R={pgd[0]['recall'] if pgd else float('nan'):.3f}  "
                  f"Decision@0.1 ASR={db[0]['asr'] if db else float('nan'):.3f}", flush=True)
        checkpoint()
        print(f"  checkpoint written ({len(rows)} rows total)", flush=True)

    df = pd.DataFrame(rows)
    print(f"\nWrote {outp}  ({len(df)} rows)")
    print("\n-- recall under PGD (undefended vs adv-trained) --")
    print(df[df.attack == 'PGD'].pivot_table(
        index='model', columns=['defense', 'eps'], values='recall').round(3).to_string())
    print("\n-- Decision-based ASR (undefended vs adv-trained) - the coherence test --")
    print(df[df.attack == 'Decision-based'].pivot_table(
        index='model', columns=['defense', 'eps'], values='asr').round(3).to_string())


if __name__ == '__main__':
    main()
