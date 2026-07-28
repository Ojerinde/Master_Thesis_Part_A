"""
Consolidate every remaining manuscript number from the single run, so all tables
are consistent and confirmable:
  - Table 1 clean F1 with 10,000-resample bootstrap 95% CI at the op-point
  - white-box FGSM/PGD table (recall, dF1) at eps=0.10
  - domain-specific / transfer mean dF1 (overview + Wilcoxon)
  - worst-case recall per model INCLUDING the decision-based attack
  - Friedman (attack heterogeneity) + Wilcoxon (DL vs classical dF1) recomputed
Prints a copy-pasteable summary. Run: PYTHONPATH=. python experiments/19_final_analysis.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
import torch
from scipy.stats import friedmanchisquare, wilcoxon

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

CLASSICAL = {'RandomForest': 'RandomForest_default', 'XGBoost': 'XGBoost_default',
             'LightGBM': 'LightGBM_default', 'GradientBoosting': 'GradientBoosting',
             'KNN': 'KNN', 'MLP': 'MLP', 'DecisionTree': 'DecisionTree', 'SVM': 'SVM'}
DL = {'CNN-1D': (CNN1DModel, 'cnn_1d'), 'LSTM': (LSTMModel, 'lstm'),
      'BiLSTM': (BiLSTMModel, 'bilstm'), 'CNN-LSTM': (CNNLSTMModel, 'cnn_lstm'),
      'Transformer': (TransformerModel, 'transformer'), 'TCN': (TCNModel, 'tcn')}
TR = 0.95


def flat(p):
    p = np.asarray(p)
    if p.ndim == 2:
        p = p[:, 1] if p.shape[1] >= 2 else p[:, 0]
    return p.ravel().astype(np.float64)


def tau_for(yv, pv):
    best, fb, br = None, 0.5, -1
    for t in np.linspace(0.01, 0.99, 197):
        r = recall_score(yv, (pv >= t).astype(int), zero_division=0)
        if r >= TR:
            best = t
        if r > br:
            br, fb = r, t
    return float(best) if best is not None else float(fb)


def load():
    # Leakage-free block-temporal split (shared load_track_splits).
    (_Xtr, Xv_e, Xte_e, _ytr, yv, yte, _fn, sc) = load_track_splits(verbose=False)
    return Xv_e.astype(np.float64), Xte_e.astype(np.float64), sc, yv, yte


def boot_ci(y, pred, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    f = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        f.append(f1_score(y[b], pred[b], zero_division=0))
    lo, hi = np.percentile(f, [2.5, 97.5])
    return float(f1_score(y, pred, zero_division=0)), float(lo), float(hi)


def main():
    Xv_e, Xte_e, sc, yv, yte = load()
    Xv_s = sc.transform(Xv_e).astype(np.float32)
    Xte_s = sc.transform(Xte_e).astype(np.float32)

    print("=== TABLE 1: clean F1 + 95% bootstrap CI (10,000) at op-point ===")
    for name, stem in CLASSICAL.items():
        m = joblib.load(CLASSICAL_MODELS / f'{stem}.joblib')
        tau = tau_for(yv, flat(m.predict_proba(Xv_e)))
        pred = (flat(m.predict_proba(Xte_e)) >= tau).astype(int)
        f, lo, hi = boot_ci(yte, pred)
        print(f'{name:16s} F1={f:.3f} CI[{lo:.3f},{hi:.3f}]')
    for name, (cls, cfg) in DL.items():
        c = get_config(cfg); c['input_dim'] = Xte_s.shape[1]
        m = cls(input_dim=Xte_s.shape[1], config=c); m.build_model()
        m.model.load_state_dict(torch.load(str(DL_MODELS / f'{cfg}.pt'), map_location='cpu'))
        m.model.eval(); m.is_trained = True
        tau = tau_for(yv, flat(m.predict_proba(Xv_s)))
        pred = (flat(m.predict_proba(Xte_s)) >= tau).astype(int)
        f, lo, hi = boot_ci(yte, pred)
        print(f'{name:16s} F1={f:.3f} CI[{lo:.3f},{hi:.3f}]')

    d = pd.read_csv(TABLES_DIR / 'adversarial_full_oppoint.csv')
    bb = pd.read_csv(TABLES_DIR / 'blackbox_boundary_all.csv')
    clean = d[d.attack == 'clean'].set_index('model')

    print("\n=== worst-case recall per model (min over ALL attacks incl decision-based) ===")
    bb20 = bb[np.isclose(bb.epsilon, 0.20)].set_index('model').asr
    rows = []
    for mdl in list(DL) + list(CLASSICAL):
        sub = d[(d.model == mdl) & (d.eps == 0.20) & (d.attack != 'clean')]
        wr = sub.recall.min(); wa = sub.loc[sub.recall.idxmin(), 'attack']
        dec_r = clean.loc[mdl, 'recall'] * (1 - bb20.get(mdl, 0))  # decision-based recall
        if dec_r < wr:
            wr, wa = dec_r, 'Decision-based'
        rows.append((mdl, clean.loc[mdl, 'family'], round(wr, 3), wa,
                     round(clean.loc[mdl, 'f1'] - sub.f1.min(), 3)))
    for r in rows:
        print(f'{r[0]:16s} {r[1]:9s} worstR={r[2]:.3f} attack={r[3]:20s} maxdF1={r[4]:+.3f}')

    print("\n=== domain-specific / transfer mean dF1 (for overview + Wilcoxon) ===")
    for atk in ['DLSA', 'SNA', 'TPA', 'PGD-Transfer', 'PGD-Transfer-Multi', 'FGSM', 'PGD']:
        for e in [0.05, 0.10, 0.20]:
            s = d[(d.attack == atk) & (d.eps == e)]
            if len(s) == 0:
                continue
            for fam in s.family.unique():
                sf = s[s.family == fam]
                dfm = (clean.loc[sf.model, 'f1'].values - sf.f1.values).mean()
                if e == 0.10 or atk in ('DLSA',):
                    print(f'{atk:20s} eps={e} {fam:9s} meandF1={dfm:+.4f}')

    print("\n=== Friedman (attack heterogeneity @0.10) ===")
    for fam, atks in [('deep', ['FGSM', 'PGD', 'DLSA', 'SNA', 'TPA']),
                      ('classical', ['PGD-Transfer', 'PGD-Transfer-Multi', 'DLSA', 'SNA', 'TPA'])]:
        mat = []
        for a in atks:
            s = d[(d.family == fam) & (d.attack == a) & (d.eps == 0.10)].set_index('model')
            mat.append((clean[clean.family == fam].f1 - s.f1).values)
        chi, p = friedmanchisquare(*mat)
        k = len(atks); n = len(mat[0]); W = chi / (n * (k - 1))
        print(f'{fam:9s} chi2={chi:.2f} p={p:.4g} W={W:.3f} (n={n},k={k})')

    print("\n=== Wilcoxon DL vs classical dF1 (paired-by-attack not valid; report Mann-Whitney per attack) ===")
    from scipy.stats import mannwhitneyu
    for atk in ['DLSA', 'SNA', 'TPA']:
        for e in [0.05, 0.10, 0.20]:
            dd = d[(d.family == 'deep') & (d.attack == atk) & (d.eps == e)]
            cc = d[(d.family == 'classical') & (d.attack == atk) & (d.eps == e)]
            ddf = clean.loc[dd.model, 'f1'].values - dd.f1.values
            cdf = clean.loc[cc.model, 'f1'].values - cc.f1.values
            try:
                u, p = mannwhitneyu(ddf, cdf, alternative='greater')
            except ValueError:
                p = np.nan
            print(f'{atk} eps={e}: DLmean={ddf.mean():+.4f} CLFmean={cdf.mean():+.4f} p={p:.4g}')


if __name__ == '__main__':
    main()
