"""
Recompute the clean-data McNemar comparability at the recall=0.95 operating
point (single run), so the "unconfounded baseline" claim is confirmable.
78 pairwise McNemar mid-p exact tests + Bonferroni-Holm. Prints the summary
that feeds tab:mcnemar_summary. Run: PYTHONPATH=. python experiments/22_mcnemar.py
"""
from pathlib import Path
import sys
import itertools
import numpy as np
import joblib
import torch
from scipy.stats import binom

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from sklearn.preprocessing import MinMaxScaler                # noqa: E402
from sklearn.model_selection import train_test_split            # noqa: E402
from sklearn.metrics import recall_score                        # noqa: E402
from data.loader import load_track_splits                       # noqa: E402
from config.paths import CLASSICAL_MODELS, DL_MODELS            # noqa: E402
from config.model_configs import get_config                     # noqa: E402
from models.deep_learning import (                               # noqa: E402
    CNN1DModel, LSTMModel, BiLSTMModel, CNNLSTMModel, TransformerModel, TCNModel)

CLASSICAL = {'RandomForest': 'RandomForest_default', 'XGBoost': 'XGBoost_default',
             'LightGBM': 'LightGBM_default', 'GradientBoosting': 'GradientBoosting',
             'KNN': 'KNN', 'MLP': 'MLP', 'DecisionTree': 'DecisionTree'}
DL = {'CNN-1D': (CNN1DModel, 'cnn_1d'), 'LSTM': (LSTMModel, 'lstm'),
      'BiLSTM': (BiLSTMModel, 'bilstm'), 'CNN-LSTM': (CNNLSTMModel, 'cnn_lstm'),
      'Transformer': (TransformerModel, 'transformer'), 'TCN': (TCNModel, 'tcn')}


def flat(p):
    p = np.asarray(p)
    return (p[:, 1] if p.ndim == 2 and p.shape[1] >= 2 else p.ravel()).astype(np.float64)


def tau_for(yv, pv):
    best, fb, br = None, 0.5, -1
    for t in np.linspace(0.01, 0.99, 197):
        r = recall_score(yv, (pv >= t).astype(int), zero_division=0)
        if r >= 0.95:
            best = t
        if r > br:
            br, fb = r, t
    return float(best) if best is not None else float(fb)


def mcnemar_midp(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    exact = min(1.0, 2 * binom.cdf(k, n, 0.5))
    midp = exact - binom.pmf(k, n, 0.5)  # mid-p correction
    return max(0.0, min(1.0, midp))


def main():
    (_Xtr, Xv_e, Xte_e, _ytr, yv, yte, fn, sc) = load_track_splits(verbose=False)
    Xv_e = Xv_e.astype(np.float64); Xte_e = Xte_e.astype(np.float64)
    Xv_s = sc.transform(Xv_e).astype(np.float32); Xte_s = sc.transform(Xte_e).astype(np.float32)

    correct = {}
    for name, stem in CLASSICAL.items():
        m = joblib.load(CLASSICAL_MODELS / f'{stem}.joblib')
        tau = tau_for(yv, flat(m.predict_proba(Xv_e)))
        correct[name] = ((flat(m.predict_proba(Xte_e)) >= tau).astype(int) == yte)
    for name, (cls, cfg) in DL.items():
        c = get_config(cfg); c['input_dim'] = Xte_s.shape[1]
        m = cls(input_dim=Xte_s.shape[1], config=c); m.build_model()
        m.model.load_state_dict(torch.load(str(DL_MODELS / f'{cfg}.pt'), map_location='cpu'))
        m.model.eval(); m.is_trained = True
        tau = tau_for(yv, flat(m.predict_proba(Xv_s)))
        correct[name] = ((flat(m.predict_proba(Xte_s)) >= tau).astype(int) == yte)

    names = list(correct)
    pvals, ors = [], []
    for a, bn in itertools.combinations(names, 2):
        ca, cb = correct[a], correct[bn]
        b = int(np.sum(ca & ~cb)); c = int(np.sum(~ca & cb))
        pvals.append(mcnemar_midp(b, c))
        ors.append((b + 0.5) / (c + 0.5))
    pvals = np.array(pvals)
    order = np.argsort(pvals)
    m = len(pvals); adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    sig = int(np.sum(adj < 0.05))
    print(f"pairs tested: {m}")
    print(f"significant (p_adj<0.05): {sig}")
    print(f"min p_adj: {adj.min():.4g}")
    print(f"max odds ratio: {max(ors):.2f}")
    print(f"median raw p: {np.median(pvals):.4g}")


if __name__ == '__main__':
    main()
