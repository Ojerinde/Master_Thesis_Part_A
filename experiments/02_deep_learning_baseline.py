"""
Deep Learning Baseline Training — trains 6 DL architectures on TEXBAT
with multiple random seeds. Loads feature engineer and scaler artifacts
saved by 01_classical_baseline.py to ensure identical data transforms.

Prerequisite: python -m experiments.01_classical_baseline
Usage:        python -m experiments.02_deep_learning_baseline
"""

from config.paths import (
    create_directories,
    DL_MODELS,
    TABLES_DIR,
    PROCESSED_DATA_DIR,
    SCALER_PATH,
)
from config.model_configs import get_config
from data.loader import load_texbat_track, load_track_splits
from evaluation.visualization import (
    plot_roc_curves,
    plot_confusion_matrices,
    plot_model_comparison,
)
from models.deep_learning import (
    CNN1DModel, LSTMModel, BiLSTMModel,
    CNNLSTMModel, TransformerModel, TCNModel,
)
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    precision_recall_curve, auc, brier_score_loss,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)
import torch
import os
import sys
import time
import pickle
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

matplotlib.use('Agg')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# Paths written by 01_classical_baseline.py
ENGINEER_PATH = PROCESSED_DATA_DIR / "feature_engineer.pkl"
RAW_FEATURE_NAMES_PATH = PROCESSED_DATA_DIR / "raw_feature_names.pkl"

# --- Experiment settings ---

N_SEEDS = 3
MIN_RECALL = 0.95

DL_CONFIGS = {
    'CNN-1D':      (CNN1DModel,       'cnn_1d'),
    'LSTM':        (LSTMModel,        'lstm'),
    'BiLSTM':      (BiLSTMModel,      'bilstm'),
    'CNN-LSTM':    (CNNLSTMModel,     'cnn_lstm'),
    'Transformer': (TransformerModel, 'transformer'),
    'TCN':         (TCNModel,         'tcn'),
}

SCALAR_KEYS = [
    'test_accuracy', 'test_precision', 'test_recall', 'test_f1',
    'test_auc_roc',  'test_pr_auc',    'test_brier',  'test_ece',
    'inference_us',  'training_time',
    'train_f1',      'overfitting_gap',
    'thresh_optimal', 'thresh_opt_f1',
    'thresh_default_f1', 'thresh_f1_gain',
]

# --- Reproducibility ---


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Metric helpers ---


def _get_proba(model, X):
    proba = model.predict_proba(X)
    return proba[:, 1].ravel().astype(np.float64) if proba.ndim == 2 else proba.ravel().astype(np.float64)


def compute_metrics(model, X, y, threshold=0.5):
    proba = _get_proba(model, X)
    preds = (proba >= threshold).astype(int)
    return {
        'accuracy':  float(accuracy_score(y, preds)),
        'precision': float(precision_score(y, preds, zero_division=0)),
        'recall':    float(recall_score(y, preds, zero_division=0)),
        'f1':        float(f1_score(y, preds, zero_division=0)),
        'auc_roc':   float(roc_auc_score(y, proba)),
    }


def calculate_pr_auc(model, X, y):
    proba = _get_proba(model, X)
    precision, recall, _ = precision_recall_curve(y, proba)
    return float(auc(recall, precision))


def calculate_brier_score(model, X, y):
    return float(brier_score_loss(y, _get_proba(model, X)))


def calculate_ece(model, X, y, n_bins=10):
    proba = _get_proba(model, X)
    try:
        prob_true, prob_pred = calibration_curve(
            y, proba, n_bins=n_bins, strategy='uniform')
        bin_sizes = np.histogram(proba, bins=n_bins, range=(0, 1))[0]
        weights = bin_sizes[:len(prob_true)].astype(float)
        total = weights.sum()
        return float(np.sum((weights/total)*np.abs(prob_true-prob_pred))) if total > 0 else float('nan')
    except Exception:
        return float('nan')


def measure_inference_time(model, X, n_repeats=3):
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        model.predict_proba(X)
        times.append((time.perf_counter()-t0)/len(X)*1e6)
    return float(np.median(times))

# --- Threshold analysis ---


def threshold_analysis(model, X, y, min_recall=MIN_RECALL, n_thresholds=181):
    proba = _get_proba(model, X)
    thresholds = np.linspace(0.05, 0.95, n_thresholds)
    records = []
    for t in thresholds:
        preds = (proba >= t).astype(int)
        tp = int(np.sum((preds == 1) & (y == 1)))
        fp = int(np.sum((preds == 1) & (y == 0)))
        fn = int(np.sum((preds == 0) & (y == 1)))
        prec = tp/(tp+fp) if (tp+fp) > 0 else 0.0
        rec = tp/(tp+fn) if (tp+fn) > 0 else 0.0
        f1_v = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        records.append(
            {'threshold': float(t), 'precision': prec, 'recall': rec, 'f1': f1_v})
    df = pd.DataFrame(records)
    feasible = df[df['recall'] >= min_recall]
    best_row = feasible.loc[feasible['f1'].idxmax(
    )] if not feasible.empty else df.loc[df['recall'].idxmax()]
    default_row = df.iloc[(df['threshold']-0.5).abs().argsort()[:1]].iloc[0]
    return {
        'threshold_df': df, 'optimal_threshold': float(best_row['threshold']),
        'optimal_precision': float(best_row['precision']), 'optimal_recall': float(best_row['recall']),
        'optimal_f1': float(best_row['f1']), 'default_precision': float(default_row['precision']),
        'default_recall': float(default_row['recall']), 'default_f1': float(default_row['f1']),
        'min_recall_constraint': min_recall, 'constraint_feasible': not feasible.empty,
    }


def plot_threshold_analysis(t_results, model_name, save_dir='dl_baseline'):
    if t_results is None:
        return
    df = t_results['threshold_df']
    opt_t = t_results['optimal_threshold']
    min_r = t_results['min_recall_constraint']
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df['threshold'], df['precision'],
            label='Precision', color='steelblue', lw=2)
    ax.plot(df['threshold'], df['recall'],
            label='Recall', color='darkorange', lw=2)
    ax.plot(df['threshold'], df['f1'], label='F1', color='seagreen', lw=2)
    ax.axhline(min_r, color='darkorange', lw=1, ls='--', alpha=0.4)
    ax.axvline(0.5, color='grey', lw=1.2, ls='--', label='Default (0.5)')
    ax.axvline(opt_t, color='red', lw=1.5, ls='--',
               label=f"Optimal t={opt_t:.2f} P={t_results['optimal_precision']:.3f} R={t_results['optimal_recall']:.3f} F1={t_results['optimal_f1']:.3f}")
    ax.set_xlabel('Decision Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Threshold Analysis — {model_name}', fontsize=13)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_dir = TABLES_DIR.parent/'figures'/save_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir/f"threshold_{model_name}.png", dpi=150)
    plt.close(fig)

# --- Single-seed training ---


def train_single_seed(model_name, ModelClass, config_name,
                      X_train, X_val, X_test, y_train, y_val, y_test, seed):
    set_seed(seed)
    try:
        config = get_config(config_name)
        config['input_dim'] = X_train.shape[1]
        model = ModelClass(input_dim=X_train.shape[1], config=config)
        model.build_model()
        t0 = time.time()
        model.train(X_train, y_train, X_val=X_val, y_val=y_val)
        training_time = time.time()-t0
        test_m = compute_metrics(model, X_test, y_test)
        pr_auc = calculate_pr_auc(model, X_test, y_test)
        brier = calculate_brier_score(model, X_test, y_test)
        ece = calculate_ece(model, X_test, y_test)
        infer_us = measure_inference_time(model, X_test)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_train), size=min(
            len(X_train), 20_000), replace=False)
        train_m = compute_metrics(model, X_train[idx], y_train[idx])
        t_results = threshold_analysis(model, X_test, y_test)
        return {
            'seed': seed, 'training_time': training_time,
            'test_accuracy': test_m['accuracy'], 'test_precision': test_m['precision'],
            'test_recall': test_m['recall'], 'test_f1': test_m['f1'],
            'test_auc_roc': test_m['auc_roc'], 'test_pr_auc': pr_auc,
            'test_brier': brier, 'test_ece': ece, 'inference_us': infer_us,
            'train_f1': train_m['f1'], 'overfitting_gap': train_m['f1']-test_m['f1'],
            'thresh_optimal': t_results['optimal_threshold'],
            'thresh_opt_f1': t_results['optimal_f1'],
            'thresh_default_f1': t_results['default_f1'],
            'thresh_f1_gain': t_results['optimal_f1']-t_results['default_f1'],
            'constraint_feasible': t_results['constraint_feasible'],
            '_model': model, '_t_results': t_results,
        }
    except Exception as exc:
        print(f"\n    Seed {seed} failed: {exc}")
        import traceback
        traceback.print_exc()
        return None

# --- Multi-seed training loop ---


def train_deep_learning_models(X_train, X_val, X_test, y_train, y_val, y_test,
                               seeds=None, save_models=True):
    if seeds is None:
        seeds = list(range(42, 42+N_SEEDS))
    print(f"\n{'='*70}\nDEEP LEARNING BASELINE  ({N_SEEDS} seeds: {seeds})\n{'='*70}")
    summary_rows = []
    best_models = {}

    for model_name, (ModelClass, config_name) in DL_CONFIGS.items():
        print(f"\n{'='*70}\nArchitecture: {model_name}\n{'='*70}")
        seed_results = []
        for seed in seeds:
            print(f"  Seed {seed} ... ", end='', flush=True)
            result = train_single_seed(model_name, ModelClass, config_name,
                                       X_train, X_val, X_test, y_train, y_val, y_test, seed)
            if result:
                seed_results.append(result)
                print(
                    f"F1={result['test_f1']:.4f}  AUC={result['test_auc_roc']:.4f}  {result['training_time']:.1f}s")
            else:
                print("FAILED")
        if not seed_results:
            print(f"  All seeds failed for {model_name}, skipping.")
            continue

        agg = {'model_name': model_name, 'n_seeds': len(seed_results)}
        for key in SCALAR_KEYS:
            vals = [r[key] for r in seed_results if key in r]
            agg[f'{key}_mean'] = float(np.mean(vals))
            agg[f'{key}_std'] = float(np.std(vals))
        agg['constraint_feasible'] = all(
            r['constraint_feasible'] for r in seed_results)

        f1_vals = [r['test_f1'] for r in seed_results]
        best_run = min(seed_results, key=lambda r: abs(
            r['test_f1']-float(np.median(f1_vals))))
        best_models[model_name] = best_run['_model']

        print(f"\n  Results ({len(seed_results)} seed(s)):")
        for metric, key in [('Accuracy', 'test_accuracy'), ('Precision', 'test_precision'),
                            ('Recall', 'test_recall'), ('F1', 'test_f1'),
                            ('AUC-ROC', 'test_auc_roc'), ('PR-AUC', 'test_pr_auc'),
                            ('Brier', 'test_brier'), ('ECE', 'test_ece'),
                            ('Overfit gap', 'overfitting_gap')]:
            print(
                f"    {metric:15s}: {agg[f'{key}_mean']:.4f} +/- {agg[f'{key}_std']:.4f}")
        print(f"    Inference  : {agg['inference_us_mean']:.2f} us/sample")

        summary_rows.append(agg)

        if save_models:
            model_path = DL_MODELS / f"{config_name}.pt"
            torch.save(best_run['_model'].model.state_dict(), str(model_path))
            print(f"  Weights saved: {model_path}  (seed={best_run['seed']})")

        plot_threshold_analysis(
            best_run['_t_results'], model_name, save_dir='dl_baseline')

    if not summary_rows:
        return pd.DataFrame(), {}
    return pd.DataFrame(summary_rows).sort_values('test_f1_mean', ascending=False).reset_index(drop=True), best_models


def main():
    print("Setting up directories...")
    create_directories()

    # ── Steps 1-4: leakage-free block-temporal split (the SAME partition as
    # 01_classical_baseline, via the shared load_track_splits), then scale for the
    # DL nets. No feature engineering: the 9 FGI observables are used directly.
    print("\n" + "="*70 + "\nSTEP 1: LOAD + BLOCK-TEMPORAL SPLIT + SCALE\n" + "="*70)
    (X_train, X_val, X_test, y_train, y_val, y_test,
     numeric_cols, scaler) = load_track_splits(verbose=True)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)
    print(f"Features: {len(numeric_cols)}  "
          f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    # ── Step 6: Train ────────────────────────────────────────────────────────
    print("\n" + "="*70 + "\nSTEP 6: DEEP LEARNING MODEL TRAINING\n" + "="*70)
    summary_df, best_models = train_deep_learning_models(
        X_train, X_val, X_test, y_train, y_val, y_test,
        seeds=list(range(42, 42+N_SEEDS)), save_models=True,
    )

    if summary_df.empty:
        print("\nNo models trained successfully.")
        return

    # ── Step 7: Save ─────────────────────────────────────────────────────────
    results_path = TABLES_DIR / "dl_baseline_results.csv"
    summary_df.to_csv(results_path, index=False)
    print(f"\nResults saved: {results_path}")

    # ── Step 8: Summary ──────────────────────────────────────────────────────
    print("\n" + "="*70 + "\nDEEP LEARNING BASELINE — RESULTS\n" + "="*70)
    primary = ['model_name', 'test_accuracy_mean', 'test_precision_mean',
               'test_recall_mean', 'test_f1_mean', 'test_f1_std',
               'test_auc_roc_mean', 'test_pr_auc_mean', 'inference_us_mean']
    print(summary_df[[c for c in primary if c in summary_df.columns]].to_string(
        index=False))

    # ── Step 9: Visualisations ───────────────────────────────────────────────
    if best_models:
        plot_roc_curves(best_models, X_test, y_test,
                        save_path="dl_baseline/roc_curves.png")
        plot_confusion_matrices(best_models, X_test,
                                y_test, save_dir="dl_baseline", prefix="dl_")
        plot_df = summary_df.rename(columns={
            'test_f1_mean': 'f1', 'test_accuracy_mean': 'accuracy',
            'test_auc_roc_mean': 'auc_roc', 'test_precision_mean': 'precision',
            'test_recall_mean': 'recall', 'training_time_mean': 'training_time',
        })
        plot_model_comparison(
            plot_df, save_path="dl_baseline/model_comparison.png")

    print("\n" + "="*70 + "\nDEEP LEARNING BASELINE COMPLETE\n" + "="*70)
    print(
        f"Best:  {summary_df.iloc[0]['model_name']}  F1={summary_df.iloc[0]['test_f1_mean']:.4f}")
    print(
        f"Worst: {summary_df.iloc[-1]['model_name']}  F1={summary_df.iloc[-1]['test_f1_mean']:.4f}")


if __name__ == "__main__":
    main()
