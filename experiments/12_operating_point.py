"""
Reviewer #2 fix — common operating point across all model families.
=====================================================================
The rejected paper tuned classical thresholds to max-F1 s.t. recall>=0.95 but
left DL models at a fixed 0.5, so clean/adversarial numbers were not comparable
across families. This script sets ONE common operating point for every model:
the decision threshold that yields clean-VALIDATION recall = 0.95 (the
safety-first sensitivity floor), then reports clean-TEST metrics at that point.

It uses the shared leakage-free block-temporal split (data.loader.load_track_splits,
deterministic) and the same 13-detector roster as every other stage: classical models
take unscaled engineered input (their Pipeline scales internally), DL models take
MinMaxScaler-scaled input.

Outputs:
  results/tables/operating_point_recall95.csv   (per-model tau + clean metrics)
  results/tables/operating_point_recall95.md

Run:
  PYTHONPATH=. python experiments/12_operating_point.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.loader import load_track_splits                # noqa: E402
from config.paths import CLASSICAL_MODELS, DL_MODELS, TABLES_DIR  # noqa: E402
from config.model_configs import get_config              # noqa: E402
from models.deep_learning import (                        # noqa: E402
    CNN1DModel, LSTMModel, BiLSTMModel,
    CNNLSTMModel, TransformerModel, TCNModel,
)

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
    """Leakage-free block-temporal split (shared load_track_splits). Classical
    models take the UNSCALED 9 observables (their Pipelines scale internally); DL
    models take the MinMaxScaler-scaled version. Same partition as 01/02."""
    (_X_train, X_val_eng, X_test_eng, _y_train, y_val, y_test,
     feat_names, scaler) = load_track_splits(verbose=False)
    X_val_eng = X_val_eng.astype(np.float64)
    X_test_eng = X_test_eng.astype(np.float64)
    X_val = scaler.transform(X_val_eng).astype(np.float32)
    X_test = scaler.transform(X_test_eng).astype(np.float32)
    return (X_val, X_test, X_val_eng, X_test_eng,
            y_val, y_test, feat_names, scaler)


def proba(model, X):
    """Flat positive-class probability, model-agnostic."""
    p = model.predict_proba(X)
    p = np.asarray(p)
    if p.ndim == 2:
        p = p[:, 1]
    return p.ravel().astype(np.float64)


def threshold_for_recall(y_val, p_val, target=TARGET_RECALL):
    """
    Largest threshold whose clean-validation recall >= target.
    Maximising tau subject to the recall floor also maximises precision at the
    operating point. Falls back to the tau giving maximum recall if the floor
    is unreachable.
    """
    taus = np.linspace(0.01, 0.99, 197)
    best_tau, best_recall_if_infeasible, best_r = None, 0.0, -1.0
    for tau in taus:
        pred = (p_val >= tau).astype(int)
        r = recall_score(y_val, pred, zero_division=0)
        if r >= target:
            best_tau = tau            # keep climbing; last qualifying = largest
        if r > best_r:
            best_r, best_recall_if_infeasible = r, tau
    if best_tau is not None:
        return float(best_tau), True
    return float(best_recall_if_infeasible), False


def threshold_for_far(y_val, p_val, target=0.05):
    """
    Smallest threshold whose clean-validation false-alarm rate <= target.
    FAR decreases as tau increases, so the smallest qualifying tau maximises
    recall among thresholds that meet the false-alarm ceiling. Second operating
    point alongside recall-0.95, so a high FAR at the safety-first point is not
    the only number a reviewer sees (both are reported; neither hides the other).
    """
    taus = np.linspace(0.01, 0.99, 197)
    for tau in taus:
        pred = (p_val >= tau).astype(int)
        far = float(((pred == 1) & (y_val == 0)).sum() / max((y_val == 0).sum(), 1))
        if far <= target:
            return float(tau), True
    return float(taus[-1]), False


def metrics_at(model, X, y, tau):
    p = proba(model, X)
    pred = (p >= tau).astype(int)
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float('nan')
    return {
        'accuracy':  float(accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall':    float(recall_score(y, pred, zero_division=0)),
        'f1':        float(f1_score(y, pred, zero_division=0)),
        'auc_roc':   auc,
        'false_alarm_rate': float(((pred == 1) & (y == 0)).sum()
                                  / max((y == 0).sum(), 1)),
    }


def main():
    print("Loading TEXBAT + preprocessing (matches 03) ...")
    (X_val, X_test, X_val_eng, X_test_eng,
     y_val, y_test, feat_names, scaler) = load_and_preprocess()
    print(f"  val {X_val.shape}  test {X_test.shape}  "
          f"test spoof rate {y_test.mean():.3f}")

    rows, rows_far = [], []

    print("\nClassical models (unscaled engineered input):")
    for disp, stem in SELECTED_CLASSICAL.items():
        path = CLASSICAL_MODELS / f"{stem}.joblib"
        if not path.exists():
            print(f"  MISSING {disp} ({stem})")
            continue
        model = joblib.load(path)
        p_val = proba(model, X_val_eng.astype(np.float64))
        tau, feasible = threshold_for_recall(y_val, p_val)
        m = metrics_at(model, X_test_eng.astype(np.float64), y_test, tau)
        m.update({'model': disp, 'family': 'classical',
                  'tau': tau, 'recall_floor_met': feasible})
        rows.append(m)
        tau_f, feas_f = threshold_for_far(y_val, p_val)
        mf = metrics_at(model, X_test_eng.astype(np.float64), y_test, tau_f)
        mf.update({'model': disp, 'family': 'classical',
                   'tau': tau_f, 'far_ceiling_met': feas_f})
        rows_far.append(mf)
        print(f"  {disp:16s} tau={tau:.3f} feasible={feasible} "
              f"testR={m['recall']:.3f} F1={m['f1']:.3f} FAR={m['false_alarm_rate']:.3f}")

    print("\nDL models (scaled input):")
    input_dim = X_test.shape[1]
    for disp, (ModelClass, cfg_name) in DL_REGISTRY.items():
        pt = DL_MODELS / f"{cfg_name}.pt"
        if not pt.exists():
            print(f"  MISSING {disp} ({cfg_name})")
            continue
        cfg = get_config(cfg_name)
        cfg['input_dim'] = input_dim
        m_ = ModelClass(input_dim=input_dim, config=cfg)
        m_.build_model()
        m_.model.load_state_dict(torch.load(str(pt), map_location='cpu'))
        m_.model.eval()
        m_.is_trained = True
        p_val = proba(m_, X_val)
        tau, feasible = threshold_for_recall(y_val, p_val)
        m = metrics_at(m_, X_test, y_test, tau)
        m.update({'model': disp, 'family': 'deep',
                  'tau': tau, 'recall_floor_met': feasible})
        rows.append(m)
        tau_f, feas_f = threshold_for_far(y_val, p_val)
        mf = metrics_at(m_, X_test, y_test, tau_f)
        mf.update({'model': disp, 'family': 'deep',
                   'tau': tau_f, 'far_ceiling_met': feas_f})
        rows_far.append(mf)
        print(f"  {disp:16s} tau={tau:.3f} feasible={feasible} "
              f"testR={m['recall']:.3f} F1={m['f1']:.3f} FAR={m['false_alarm_rate']:.3f}")

    df = pd.DataFrame(rows)[
        ['model', 'family', 'tau', 'recall_floor_met', 'recall',
         'false_alarm_rate', 'precision', 'f1', 'accuracy', 'auc_roc']]
    out_csv = TABLES_DIR / 'operating_point_recall95.csv'
    df.to_csv(out_csv, index=False)
    with open(TABLES_DIR / 'operating_point_recall95.md', 'w') as f:
        f.write("# Common operating point: clean-validation recall = 0.95\n\n")
        f.write(df.round(4).to_markdown(index=False))
        f.write("\n")
    print(f"\nWrote {out_csv}")
    print(df.round(4).to_string(index=False))

    df_far = pd.DataFrame(rows_far)[
        ['model', 'family', 'tau', 'far_ceiling_met', 'recall',
         'false_alarm_rate', 'precision', 'f1', 'accuracy', 'auc_roc']]
    out_csv_far = TABLES_DIR / 'operating_point_far05.csv'
    df_far.to_csv(out_csv_far, index=False)
    with open(TABLES_DIR / 'operating_point_far05.md', 'w') as f:
        f.write("# Second operating point: clean-validation false-alarm rate <= 0.05\n\n")
        f.write(df_far.round(4).to_markdown(index=False))
        f.write("\n")
    print(f"\nWrote {out_csv_far}")
    print(df_far.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
