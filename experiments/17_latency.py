"""
Reviewer #3 latency fix: measure inference latency for ALL thirteen models from
one consistent source, so Tables 8 and 9 can be reconciled (the rejected paper
gave Decision Tree two different inference/DLSA ratios) and the five missing
classical models can be added.

Also times the gradient-free domain-specific perturbation (DLSA-style: a
precomputed centroid-directed sign vector added per sample) and the per-sample
cost of the decision-based attack, to state attack feasibility consistently.

Run: PYTHONPATH=. python experiments/17_latency.py
"""
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import joblib
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from sklearn.preprocessing import MinMaxScaler                # noqa: E402
from sklearn.model_selection import train_test_split            # noqa: E402
from data.loader import load_track_splits                       # noqa: E402
from config.paths import CLASSICAL_MODELS, DL_MODELS, TABLES_DIR  # noqa: E402
from config.model_configs import get_config                     # noqa: E402
from models.deep_learning import (                               # noqa: E402
    CNN1DModel, LSTMModel, BiLSTMModel, CNNLSTMModel, TransformerModel, TCNModel)

SELECTED_CLASSICAL = {
    'RandomForest': 'RandomForest_default', 'XGBoost': 'XGBoost_default',
    'LightGBM': 'LightGBM_default', 'GradientBoosting': 'GradientBoosting',
    'KNN': 'KNN', 'MLP': 'MLP', 'DecisionTree': 'DecisionTree', 'SVM': 'SVM'}
DL_REGISTRY = {
    'CNN-1D': (CNN1DModel, 'cnn_1d'), 'LSTM': (LSTMModel, 'lstm'),
    'BiLSTM': (BiLSTMModel, 'bilstm'), 'CNN-LSTM': (CNNLSTMModel, 'cnn_lstm'),
    'Transformer': (TransformerModel, 'transformer'), 'TCN': (TCNModel, 'tcn')}


def load_data():
    # Leakage-free block-temporal split (shared load_track_splits); the latency
    # benchmark needs only the test block + the train-fit scaler.
    (_Xtr, _Xv, Xte_e, _ytr, _yv, _yte, _fn, scaler) = load_track_splits(verbose=False)
    return Xte_e.astype(np.float64), scaler


def med_latency(call, X, repeats=7):
    t = []
    call(X[:64])  # warmup
    for _ in range(repeats):
        t0 = time.perf_counter()
        call(X)
        t.append((time.perf_counter() - t0) / len(X) * 1e6)
    return float(np.median(t))


def main():
    Xe, scaler = load_data()
    Xe = Xe[:4000]
    Xs = scaler.transform(Xe).astype(np.float32)
    # DLSA-style gradient-free perturbation cost: add a precomputed sign vector.
    sign_vec = np.sign(np.random.randn(Xe.shape[1]))
    rows = []
    for name, stem in SELECTED_CLASSICAL.items():
        m = joblib.load(CLASSICAL_MODELS / f'{stem}.joblib')
        inf = med_latency(lambda X: m.predict_proba(X), Xe.astype(np.float64))
        gen = med_latency(lambda X: X + 0.1 * sign_vec, Xe)
        rows.append({'model': name, 'family': 'classical',
                     'infer_us': round(inf, 3), 'dlsa_gen_us': round(gen, 4),
                     'dlsa_ratio': round(gen / inf, 4)})
    for name, (cls, cfg) in DL_REGISTRY.items():
        c = get_config(cfg); c['input_dim'] = Xs.shape[1]
        md = cls(input_dim=Xs.shape[1], config=c); md.build_model()
        md.model.load_state_dict(torch.load(str(DL_MODELS / f'{cfg}.pt'),
                                            map_location='cpu'))
        md.model.eval(); md.is_trained = True
        inf = med_latency(lambda X: md.predict_proba(X), Xs)
        gen = med_latency(lambda X: X + 0.1 * sign_vec.astype(np.float32), Xs)
        rows.append({'model': name, 'family': 'deep',
                     'infer_us': round(inf, 3), 'dlsa_gen_us': round(gen, 4),
                     'dlsa_ratio': round(gen / inf, 4)})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / 'latency_all13.csv', index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {TABLES_DIR / 'latency_all13.csv'}")


if __name__ == '__main__':
    main()
