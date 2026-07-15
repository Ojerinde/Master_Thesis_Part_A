"""
Reviewer #1 fix (part B) — multi-surrogate transfer attack on classical models.
=========================================================================
The rejected paper crafted transfer adversarial examples on a SINGLE CNN-1D
surrogate (an admitted limitation) and read the resulting low classical ASR as
robustness. Here we craft PGD adversarial examples on an ENSEMBLE of DL
surrogates (CNN-1D + LSTM + TCN) by averaging their logits before back-prop
(Liu et al. 2017, "Delving into transferable adversarial examples"), which is a
much stronger transfer attack, then evaluate the transfer onto the seven
classical targets at the common recall=0.95 operating point (reviewer #2).

We report, per target / epsilon / surrogate-set, the Attack Success Rate on
correctly-detected SPOOF points (the safety-critical missed-detection direction)
and the change in recall/F1. Single-surrogate (CNN-1D) is included as the
baseline so the ensemble gain is explicit.

Run: PYTHONPATH=. python experiments/14_multisurrogate_transfer.py [--quick]
"""
from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.preprocessing import StandardScaler                # noqa: E402
from sklearn.model_selection import train_test_split            # noqa: E402
from sklearn.metrics import recall_score, f1_score              # noqa: E402
from data.loader import load_track_splits                       # noqa: E402
from config.paths import CLASSICAL_MODELS, DL_MODELS, TABLES_DIR  # noqa: E402
from config.model_configs import get_config                     # noqa: E402
from models.deep_learning import CNN1DModel, LSTMModel, TCNModel  # noqa: E402
from attacks.pgd import PGDAttack                                # noqa: E402
from utils.gnss_constraints import GNSSConstraintEnforcer         # noqa: E402

EPSILONS = [0.05, 0.10, 0.20]
TARGET_RECALL = 0.95
SURROGATES = {'CNN-1D': (CNN1DModel, 'cnn_1d'),
              'LSTM':   (LSTMModel,  'lstm'),
              'TCN':    (TCNModel,   'tcn')}
SELECTED_CLASSICAL = {
    'RandomForest':     'RandomForest_default',
    'XGBoost':          'XGBoost_default',
    'LightGBM':         'LightGBM_default',
    'GradientBoosting': 'GradientBoosting',
    'KNN':              'KNN',
    'MLP':              'MLP',
    'DecisionTree':     'DecisionTree',
}


class EnsembleLogits(nn.Module):
    """Mean-logit ensemble of DL surrogate nn.Modules (Liu et al. 2017)."""
    def __init__(self, modules):
        super().__init__()
        self.mods = nn.ModuleList(modules)

    def forward(self, x):
        return torch.stack([m(x) for m in self.mods], dim=0).mean(dim=0)


class _Wrapper:
    """Minimal model_wrapper (.model) for PGDAttack."""
    def __init__(self, module):
        self.model = module


def load_and_preprocess():
    # Leakage-free block-temporal split (shared load_track_splits); the 9 FGI
    # observables used directly. X_* scaled for DL; *_eng unscaled for classical.
    (X_train_eng, X_val_eng, X_test_eng, _ytr, y_val, y_test,
     feat_names, scaler) = load_track_splits(verbose=False)
    X_train = scaler.transform(X_train_eng).astype(np.float32)
    X_val = scaler.transform(X_val_eng).astype(np.float32)
    X_test = scaler.transform(X_test_eng).astype(np.float32)
    return (X_train, X_val, X_test,
            X_val_eng.astype(np.float64), X_test_eng.astype(np.float64),
            y_val, y_test, feat_names, scaler)


def pos_proba(model, X):
    p = np.asarray(model.predict_proba(X))
    return (p[:, 1] if p.ndim == 2 else p).ravel().astype(np.float64)


def threshold_for_recall(y_val, p_val, target=TARGET_RECALL):
    best_tau, fallback, best_r = None, 0.5, -1.0
    for tau in np.linspace(0.01, 0.99, 197):
        r = recall_score(y_val, (p_val >= tau).astype(int), zero_division=0)
        if r >= target:
            best_tau = tau
        if r > best_r:
            best_r, fallback = r, tau
    return float(best_tau) if best_tau is not None else float(fallback)


def load_surrogate_module(ModelClass, cfg_name, input_dim):
    cfg = get_config(cfg_name)
    cfg['input_dim'] = input_dim
    m = ModelClass(input_dim=input_dim, config=cfg)
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
    (X_train, X_val, X_test, X_val_eng, X_test_eng,
     y_val, y_test, feat_names, scaler) = load_and_preprocess()
    input_dim = X_test.shape[1]

    enforcer_dl = GNSSConstraintEnforcer(feat_names)
    enforcer_dl.fit(X_train, feat_names)

    # Shared crafting subsample (stratified) for efficiency + comparability.
    n_craft = 800 if args.quick else 4000
    rng = np.random.default_rng(42)
    if len(y_test) > n_craft:
        idx = np.concatenate([
            rng.choice(np.where(y_test == c)[0],
                       size=int(round(n_craft * (y_test == c).mean())),
                       replace=False)
            for c in (0, 1)])
    else:
        idx = np.arange(len(y_test))
    Xc, yc = X_test[idx], y_test[idx]
    Xc_eng = X_test_eng[idx]

    # Pre-load classical targets + their recall-0.95 thresholds.
    targets = {}
    for disp, stem in SELECTED_CLASSICAL.items():
        model = joblib.load(CLASSICAL_MODELS / f"{stem}.joblib")
        tau = threshold_for_recall(y_val, pos_proba(model, X_val_eng))
        targets[disp] = (model, tau)

    # Surrogate modules.
    surr = {name: load_surrogate_module(cls, cfg, input_dim)
            for name, (cls, cfg) in SURROGATES.items()}
    surrogate_sets = {
        'single(CNN-1D)': _Wrapper(surr['CNN-1D'].model),
        'ensemble(CNN-1D+LSTM+TCN)': _Wrapper(
            EnsembleLogits([surr['CNN-1D'].model, surr['LSTM'].model,
                            surr['TCN'].model])),
    }

    eps_list = [0.10] if args.quick else EPSILONS
    rows = []
    for set_name, wrap in surrogate_sets.items():
        for eps in eps_list:
            pgd = PGDAttack(wrap, epsilon=eps, num_iter=40, random_start=True,
                            gnss_enforcer=enforcer_dl)
            Xadv = pgd.generate(Xc, yc)                       # scaled space
            Xadv_eng = scaler.inverse_transform(Xadv).astype(np.float64)
            for disp, (model, tau) in targets.items():
                p_cl = pos_proba(model, Xc_eng)
                p_ad = pos_proba(model, Xadv_eng)
                pred_cl = (p_cl >= tau).astype(int)
                pred_ad = (p_ad >= tau).astype(int)
                cs = (yc == 1) & (pred_cl == 1)              # correct spoof
                asr = (float(np.mean(pred_ad[cs] == 0))
                       if cs.sum() else float('nan'))
                rows.append({
                    'target': disp, 'surrogate_set': set_name, 'epsilon': eps,
                    'tau': round(tau, 4), 'n_correct_spoof': int(cs.sum()),
                    'asr_spoof_to_auth': round(asr, 4),
                    'recall_clean': round(recall_score(yc, pred_cl, zero_division=0), 4),
                    'recall_adv': round(recall_score(yc, pred_ad, zero_division=0), 4),
                    'f1_clean': round(f1_score(yc, pred_cl, zero_division=0), 4),
                    'f1_adv': round(f1_score(yc, pred_ad, zero_division=0), 4),
                })
            print(f"  {set_name} eps={eps}: crafted {Xadv.shape[0]} pts")

    df = pd.DataFrame(rows)
    tag = '_quick' if args.quick else ''
    out = TABLES_DIR / f'multisurrogate_transfer{tag}.csv'
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    # Quick view: mean ASR gain single -> ensemble.
    piv = df.pivot_table(index='target', columns='surrogate_set',
                         values='asr_spoof_to_auth', aggfunc='mean')
    print(piv.round(3).to_string())


if __name__ == '__main__':
    main()
