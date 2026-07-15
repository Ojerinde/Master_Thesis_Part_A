"""
Generalization evaluation (audit #10 / TEXBAT-standard 3-view protocol).
Two held-out protocols, each RETRAINING every model on the reduced training set
so nothing about the held-out fold leaks into training:

  cross-scenario : leave one spoofed scenario out (train on cleanStatic + the
                   other spoofed scenarios, test on the held-out spoofed one).
                   Generalization to an unseen ATTACK TYPE. Needs >=2 spoofed
                   scenarios in the corpus (runs fully once ds2/ds3 are added;
                   folds whose train or test has a single class are skipped).
  leave-PRN-out  : leave one PRN out (train on the other PRNs, test on it).
                   Generalization across SATELLITES.

Within each fold's training rows a block-temporal validation slice sets the
common recall=0.95 operating point (same rule as 12_operating_point). Detection
recall / F1 at that tau are reported on the held-out test rows.

Output: results/tables/generalization.csv   (feeds figure F8)
Run:    PYTHONPATH=. python experiments/23_generalization.py [--classical-only]
"""
from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from sklearn.preprocessing import StandardScaler                       # noqa: E402
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier                     # noqa: E402
from sklearn.neural_network import MLPClassifier                       # noqa: E402
from sklearn.tree import DecisionTreeClassifier                        # noqa: E402
from sklearn.pipeline import Pipeline                                  # noqa: E402
from sklearn.metrics import recall_score, f1_score                     # noqa: E402
from imblearn.pipeline import Pipeline as ImbPipeline                  # noqa: E402
from imblearn.over_sampling import SMOTE                               # noqa: E402
from xgboost import XGBClassifier                                      # noqa: E402
from lightgbm import LGBMClassifier                                    # noqa: E402
from data.loader import load_texbat_track                             # noqa: E402
from config.paths import TABLES_DIR                                    # noqa: E402
from config.model_configs import get_config                           # noqa: E402
from models.deep_learning import (                                     # noqa: E402
    CNN1DModel, LSTMModel, BiLSTMModel, CNNLSTMModel, TransformerModel, TCNModel)

TARGET_RECALL = 0.95
DL = {'CNN-1D': (CNN1DModel, 'cnn_1d'), 'LSTM': (LSTMModel, 'lstm'),
      'BiLSTM': (BiLSTMModel, 'bilstm'), 'CNN-LSTM': (CNNLSTMModel, 'cnn_lstm'),
      'Transformer': (TransformerModel, 'transformer'), 'TCN': (TCNModel, 'tcn')}


def classical_zoo():
    """Same 7 architectures / hyperparameters as the main run (RandomForest uses
    the SMOTE pipeline; Pipelines scale internally so they take unscaled X)."""
    def std(e):
        return Pipeline([('sc', StandardScaler()), ('m', e)])

    def smote(e):
        return ImbPipeline([('sc', StandardScaler()),
                            ('sm', SMOTE(random_state=42, k_neighbors=5)), ('m', e)])
    return {
        'RandomForest':     smote(RandomForestClassifier(**get_config('random_forest'))),
        'XGBoost':          std(XGBClassifier(**get_config('xgboost'))),
        'LightGBM':         std(LGBMClassifier(**get_config('lightgbm'))),
        'GradientBoosting': std(GradientBoostingClassifier(**get_config('gradient_boosting'))),
        'KNN':              std(KNeighborsClassifier(**get_config('knn'))),
        'MLP':              std(MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000,
                                              random_state=42, early_stopping=True)),
        'DecisionTree':     std(DecisionTreeClassifier(**get_config('decision_tree'))),
    }


def tau_for(yv, pv):
    best, fb, br = None, 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 197):
        r = recall_score(yv, (pv >= t).astype(int), zero_division=0)
        if r >= TARGET_RECALL:
            best = t
        if r > br:
            br, fb = r, t
    return float(best) if best is not None else float(fb)


def pos(p):
    p = np.asarray(p)
    return (p[:, 1] if p.ndim == 2 and p.shape[1] >= 2 else p.ravel()).astype(np.float64)


def block_val(idx_train, df, val_frac=0.15, purge=20):
    """Carve a block-temporal validation slice from the fold's training rows."""
    va = []
    for _, g in df.loc[idx_train].groupby(['scenario', 'prn', 'segment'], sort=False):
        gi = g.sort_values('t_sec').index.to_numpy()
        k = int(round(len(gi) * (1 - val_frac)))
        va.append(gi[k + purge:])
    va = np.concatenate(va) if va else np.array([], dtype=int)
    tr = np.setdiff1d(idx_train, va)
    return tr, va


def eval_fold(df, feats, tr_idx, te_idx, with_dl):
    tr, va = block_val(tr_idx, df)
    Xtr = df.loc[tr, feats].values.astype(np.float64); ytr = df.loc[tr, 'label'].values.astype(int)
    Xva = df.loc[va, feats].values.astype(np.float64); yva = df.loc[va, 'label'].values.astype(int)
    Xte = df.loc[te_idx, feats].values.astype(np.float64); yte = df.loc[te_idx, 'label'].values.astype(int)
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2 or len(np.unique(yva)) < 2:
        return []   # degenerate fold (single-class train/val/test) -> skip
    sc = StandardScaler().fit(Xtr)
    rows = []
    for name, mdl in classical_zoo().items():
        mdl.fit(Xtr, ytr)
        tau = tau_for(yva, pos(mdl.predict_proba(Xva)))
        pte = (pos(mdl.predict_proba(Xte)) >= tau).astype(int)
        rows.append(dict(model=name, family='classical',
                         recall=float(recall_score(yte, pte, zero_division=0)),
                         f1=float(f1_score(yte, pte, zero_division=0)), tau=tau))
    if with_dl:
        import torch  # noqa: F401
        Xtr_f = sc.transform(Xtr).astype(np.float32)
        Xva_f = sc.transform(Xva).astype(np.float32)
        Xte_f = sc.transform(Xte).astype(np.float32)
        for name, (cls, cfg) in DL.items():
            c = get_config(cfg); c['input_dim'] = Xtr_f.shape[1]
            m = cls(input_dim=Xtr_f.shape[1], config=c); m.build_model()
            m.train(Xtr_f, ytr, X_val=Xva_f, y_val=yva)
            tau = tau_for(yva, pos(m.predict_proba(Xva_f)))
            pte = (pos(m.predict_proba(Xte_f)) >= tau).astype(int)
            rows.append(dict(model=name, family='deep',
                             recall=float(recall_score(yte, pte, zero_division=0)),
                             f1=float(f1_score(yte, pte, zero_division=0)), tau=tau))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classical-only', action='store_true',
                    help='skip the (GPU-heavy) per-fold DL retraining')
    args = ap.parse_args()

    df, feats = load_texbat_track(verbose=True, validate=True)
    df = df.reset_index(drop=True)
    with_dl = not args.classical_only

    # Reproducibility: sklearn folds are already fixed by random_state=42; the DL
    # folds need explicit torch seeding so the result is stable across machines
    # (e.g. a Kaggle GPU run reproduces). cudnn.deterministic avoids nondeterministic
    # convolution kernels; benchmark off avoids autotuner-driven variation.
    if with_dl:
        import torch
        import random as _random
        _random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"  DL device: {'cuda' if torch.cuda.is_available() else 'cpu'}; "
              f"seeds fixed (42), cudnn deterministic")

    out = []

    spoofed = sorted(s for s in df['scenario'].unique() if s != 'cleanstatic')
    for hold in spoofed:
        te = df.index[df['scenario'] == hold].to_numpy()
        tr = df.index[df['scenario'] != hold].to_numpy()
        print(f"\n[cross-scenario] holdout={hold}  train={len(tr):,} test={len(te):,}")
        for r in eval_fold(df, feats, tr, te, with_dl):
            r.update(protocol='cross_scenario', holdout=hold); out.append(r)

    for hold in sorted(df['prn'].unique()):
        te = df.index[df['prn'] == hold].to_numpy()
        tr = df.index[df['prn'] != hold].to_numpy()
        print(f"[leave-PRN] holdout PRN={hold}  train={len(tr):,} test={len(te):,}")
        for r in eval_fold(df, feats, tr, te, with_dl):
            r.update(protocol='leave_prn', holdout=int(hold)); out.append(r)

    res = pd.DataFrame(out)
    outp = TABLES_DIR / 'generalization.csv'
    res.to_csv(outp, index=False)
    print(f"\nWrote {outp}  ({len(res)} rows)")
    if not res.empty:
        print(res.groupby(['protocol', 'family'])[['recall', 'f1']].mean().round(4).to_string())


if __name__ == '__main__':
    main()
