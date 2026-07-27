"""
Reviewer #1 fix (part A) — strong model-specific decision-based attack on the
classical models, so their robustness is not an artifact of only ever seeing
transfer + gradient-free heuristics.
=========================================================================
We use a decision-based Boundary Attack (Brendel et al. 2018; same query model
as HopSkipJump, Chen et al. 2019 — labels only, no gradients, no surrogate). ART
has no Python-3.14 wheel and the cleverhans HopSkipJump makes O(1e4) queries per
sample, which is infeasible against 200-tree ensembles. We therefore use a
VECTORIZED boundary search that evaluates all N points in a single predict call
per binary-search step, giving the same minimal-perturbation decision-based
metric ~1000x faster.

Per correctly-detected SPOOF point (the safety-critical missed-detection
direction), we find the smallest L-inf perturbation, in the min-max-normalized
[0,1] feature space, that flips it to "authentic" at the model's recall=0.95
operating threshold. We report Attack Success Rate within each epsilon budget
{0.05,0.10,0.20} and the median minimal L-inf.

Working in [0,1] also fixes the parameter-consistency issue (reviewer #3): the
realized L-inf is directly comparable to the epsilon budgets the paper defines
in that same [0,1] space.

Run: PYTHONPATH=. python experiments/13_blackbox_attacks.py [--quick]
"""
from pathlib import Path
import sys
import argparse
import time
import numpy as np
import pandas as pd
import joblib

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.model_selection import train_test_split            # noqa: E402
from sklearn.preprocessing import MinMaxScaler                # noqa: E402
from sklearn.metrics import recall_score                        # noqa: E402
import torch                                                    # noqa: E402
import joblib as _joblib                                        # noqa: E402,F401
from data.loader import load_track_splits                       # noqa: E402
from config.paths import CLASSICAL_MODELS, DL_MODELS, TABLES_DIR  # noqa: E402
from config.model_configs import get_config                     # noqa: E402
from models.deep_learning import (                               # noqa: E402
    CNN1DModel, LSTMModel, BiLSTMModel,
    CNNLSTMModel, TransformerModel, TCNModel,
)

EPSILONS = [0.05, 0.10, 0.20]
TARGET_RECALL = 0.95

SELECTED_CLASSICAL = {
    'RandomForest':     'RandomForest_default',
    'XGBoost':          'XGBoost_default',
    'LightGBM':         'LightGBM_default',
    'GradientBoosting': 'GradientBoosting',
    'KNN':              'KNN',
    'MLP':              'MLP',
    'DecisionTree':     'DecisionTree',
}
DL_REGISTRY = {
    'CNN-1D':      (CNN1DModel,       'cnn_1d'),
    'LSTM':        (LSTMModel,        'lstm'),
    'BiLSTM':      (BiLSTMModel,      'bilstm'),
    'CNN-LSTM':    (CNNLSTMModel,     'cnn_lstm'),
    'Transformer': (TransformerModel, 'transformer'),
    'TCN':         (TCNModel,         'tcn'),
}


def load_and_preprocess():
    # Leakage-free block-temporal split (shared load_track_splits); the 9 FGI
    # observables used directly. Scaler (train-fit) used for the DL proba path.
    (X_train_eng, X_val_eng, X_test_eng, _ytr, y_val, y_test,
     _fn, scaler) = load_track_splits(verbose=False)
    return (X_train_eng.astype(np.float64), X_val_eng.astype(np.float64),
            X_test_eng.astype(np.float64), y_val, y_test, scaler)


def _flat_pos(p):
    p = np.asarray(p)
    if p.ndim == 2:
        p = p[:, 1] if p.shape[1] >= 2 else p[:, 0]
    return p.ravel().astype(np.float64)


def make_classical_proba(model):
    """proba on ENGINEERED (unscaled) input."""
    return lambda X_eng: _flat_pos(model.predict_proba(X_eng))


def make_dl_proba(dl, scaler):
    """proba on ENGINEERED input (scales internally for the DL model)."""
    def f(X_eng):
        Xs = scaler.transform(X_eng).astype(np.float32)
        return _flat_pos(dl.predict_proba(Xs))
    return f


def threshold_for_recall(y_val, p_val, target=TARGET_RECALL):
    best_tau, fallback, best_r = None, 0.5, -1.0
    for tau in np.linspace(0.01, 0.99, 197):
        r = recall_score(y_val, (p_val >= tau).astype(int), zero_division=0)
        if r >= target:
            best_tau = tau
        if r > best_r:
            best_r, fallback = r, tau
    return float(best_tau) if best_tau is not None else float(fallback)


def boundary_attack(proba_eng, Xs, anchors, tau, fmin, span,
                    n_line=18, n_clip=18):
    """
    Vectorized decision-based L-inf boundary attack in [0,1] space.

    proba_eng : callable, engineered (unscaled) input -> positive-class proba.
    Xs        : (N,d) normalized spoof sources (classified spoof).
    anchors   : (K,d) normalized authentic-classified points.
    Returns   : best_adv (N,d), best_linf (N,) minimal L-inf that flips to auth.
    Each binary-search step evaluates all N points in one predict call.
    """
    def is_auth(Xn):
        return proba_eng(Xn * span + fmin) < tau

    N = Xs.shape[0]
    best_linf = np.full(N, np.inf)
    best_adv = Xs.copy()

    for a in anchors:
        direction = a[None, :] - Xs                     # (N,d), anchor is auth
        # 1) line search: smallest t s.t. Xs + t*dir is authentic.
        t_lo, t_hi = np.zeros(N), np.ones(N)
        for _ in range(n_line):
            t_mid = 0.5 * (t_lo + t_hi)
            auth = is_auth(Xs + t_mid[:, None] * direction)
            t_hi = np.where(auth, t_mid, t_hi)
            t_lo = np.where(auth, t_lo, t_mid)
        delta = t_hi[:, None] * direction               # authentic-side delta
        # 2) L-inf clip refinement: smallest c s.t. clip(delta,-c,c) stays auth.
        c_lo = np.zeros(N)
        c_hi = np.max(np.abs(delta), axis=1) + 1e-9
        for _ in range(n_clip):
            c_mid = 0.5 * (c_lo + c_hi)
            dc = np.clip(delta, -c_mid[:, None], c_mid[:, None])
            auth = is_auth(np.clip(Xs + dc, 0.0, 1.0))
            c_hi = np.where(auth, c_mid, c_hi)
            c_lo = np.where(auth, c_lo, c_mid)
        dc = np.clip(delta, -c_hi[:, None], c_hi[:, None])
        adv = np.clip(Xs + dc, 0.0, 1.0)
        better = c_hi < best_linf
        best_linf = np.where(better, c_hi, best_linf)
        best_adv = np.where(better[:, None], adv, best_adv)
    return best_adv, best_linf


def run_target(name, family, proba_eng, val_eng, test_eng, y_val, y_test,
               fmin, span, n_samples, n_anchors, seed=42):
    tau = threshold_for_recall(y_val, proba_eng(val_eng))

    def to_norm(X):
        return np.clip((X - fmin) / span, 0.0, 1.0)

    p_test = proba_eng(test_eng)
    rng = np.random.default_rng(seed)
    correct_spoof = np.where((y_test == 1) & (p_test >= tau))[0]
    if len(correct_spoof) == 0:
        return [], np.array([])
    idx = (rng.choice(correct_spoof, size=n_samples, replace=False)
           if len(correct_spoof) > n_samples else correct_spoof)
    Xs = to_norm(test_eng[idx])
    n = len(idx)

    auth_pool = np.where((y_test == 0) & (p_test < tau))[0]
    a_idx = rng.choice(auth_pool, size=min(n_anchors, len(auth_pool)),
                       replace=False)
    anchors = to_norm(test_eng[a_idx])

    t0 = time.perf_counter()
    adv, linf = boundary_attack(proba_eng, Xs, anchors, tau, fmin, span)
    gen_time = time.perf_counter() - t0

    flipped = proba_eng(adv * span + fmin) < tau
    rows = []
    for eps in list(EPSILONS) + [float('inf')]:
        within = linf <= eps + 1e-9
        asr = float(np.mean(flipped & within)) if n else float('nan')
        rows.append({'model': name, 'family': family, 'attack': 'BoundaryAttack',
                     'epsilon': eps, 'n': int(n), 'tau': round(tau, 4),
                     'asr': round(asr, 4),
                     'median_min_linf': round(float(np.median(linf)), 4),
                     'gen_time_s': round(gen_time, 1)})
    return rows, linf


def load_dl(name, cfg_name, input_dim):
    cfg = get_config(cfg_name)
    cfg['input_dim'] = input_dim
    m = DL_REGISTRY[name][0](input_dim=input_dim, config=cfg)
    m.build_model()
    m.model.load_state_dict(torch.load(str(DL_MODELS / f"{cfg_name}.pt"),
                                       map_location='cpu'))
    m.model.eval()
    m.is_trained = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()

    print("Loading TEXBAT + preprocessing ...")
    (X_train_eng, X_val_eng, X_test_eng, y_val, y_test,
     scaler) = load_and_preprocess()
    fmin = X_train_eng.min(axis=0)
    span = np.maximum(X_train_eng.max(axis=0) - fmin, 1e-12)
    input_dim = X_test_eng.shape[1]

    # Build proba_eng callables for every model (same [0,1] engineered space).
    targets = []
    if args.quick:
        m = _joblib.load(CLASSICAL_MODELS / 'RandomForest_default.joblib')
        targets.append(('RandomForest', 'classical', make_classical_proba(m)))
        targets.append(('CNN-1D', 'deep',
                        make_dl_proba(load_dl('CNN-1D', 'cnn_1d', input_dim), scaler)))
        n_samples, n_anchors, out_tag = 40, 5, '_quick'
    else:
        for disp, stem in SELECTED_CLASSICAL.items():
            m = _joblib.load(CLASSICAL_MODELS / f'{stem}.joblib')
            targets.append((disp, 'classical', make_classical_proba(m)))
        for disp, (_, cfg_name) in DL_REGISTRY.items():
            targets.append((disp, 'deep',
                            make_dl_proba(load_dl(disp, cfg_name, input_dim), scaler)))
        n_samples, n_anchors, out_tag = 500, 12, ''

    all_rows = []
    persample = []
    for name, family, proba_eng in targets:
        print(f"\n=== BoundaryAttack: {name} ({family}, N={n_samples}) ===")
        rows, linf = run_target(name, family, proba_eng, X_val_eng, X_test_eng,
                                y_val, y_test, fmin, span, n_samples, n_anchors)
        for r in rows:
            print(f"  eps={r['epsilon']}: ASR={r['asr']} "
                  f"med_min_linf={r['median_min_linf']} t={r['gen_time_s']}s")
        all_rows.extend(rows)
        for v in linf:
            persample.append({'model': name, 'family': family, 'min_linf': float(v)})

    df = pd.DataFrame(all_rows)
    out = TABLES_DIR / f'blackbox_boundary_all{out_tag}.csv'
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    ps = pd.DataFrame(persample)
    psout = TABLES_DIR / f'blackbox_boundary_persample{out_tag}.csv'
    ps.to_csv(psout, index=False)
    print(f"Wrote {psout}  ({len(ps)} per-sample rows)")


if __name__ == '__main__':
    main()
