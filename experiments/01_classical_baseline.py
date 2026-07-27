"""
Classical Baseline Training — trains 17 model variants (7 base classifiers
x default/balanced/SMOTE + single-variant models) with 5-fold CV,
hyperparameter tuning, probability calibration, and threshold optimization.

Usage:
    python -m experiments.01_classical_baseline
"""

from evaluation.visualization import (
    plot_roc_curves,
    plot_confusion_matrices,
    plot_model_comparison,
)
from evaluation.metrics import evaluate_model, compare_models
from data.loader import load_texbat_track, load_track_splits
from config.model_configs import get_config
from config.paths import create_directories, CLASSICAL_MODELS, TABLES_DIR, PROCESSED_DATA_DIR, SCALER_PATH
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, GridSearchCV
)
from sklearn.metrics import precision_recall_curve, auc, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import os
import sys
import time
import pickle
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

matplotlib.use('Agg')


os.environ['LIGHTGBM_VERBOSITY'] = '-1'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings(
    'ignore', message='X does not have valid feature names')


# Artifact paths — loaded by 02_deep_learning_baseline.py
ENGINEER_PATH = PROCESSED_DATA_DIR / "feature_engineer.pkl"
RAW_FEATURE_NAMES_PATH = PROCESSED_DATA_DIR / "raw_feature_names.pkl"

# --- Hyperparameter grids ---

HYPERPARAM_GRIDS = {
    "RandomForest_default": {
        "model__n_estimators":     [100, 200],
        "model__max_depth":        [10, 20, None],
        "model__min_samples_leaf": [1, 5],
    },
    "RandomForest_balanced": {
        "model__n_estimators":     [100, 200],
        "model__max_depth":        [10, 20, None],
        "model__min_samples_leaf": [1, 5],
    },
    "RandomForest_SMOTE": {
        "model__n_estimators":     [100, 200],
        "model__max_depth":        [10, 20, None],
        "model__min_samples_leaf": [1, 5],
    },
    "XGBoost_default": {
        "model__n_estimators":  [100, 200],
        "model__max_depth":     [4, 6, 8],
        "model__learning_rate": [0.05, 0.1],
    },
    "XGBoost_balanced": {
        "model__n_estimators":  [100, 200],
        "model__max_depth":     [4, 6, 8],
        "model__learning_rate": [0.05, 0.1],
    },
    "XGBoost_SMOTE": {
        "model__n_estimators":  [100, 200],
        "model__max_depth":     [4, 6, 8],
        "model__learning_rate": [0.05, 0.1],
    },
    "LightGBM_default": {
        "model__n_estimators":  [100, 200],
        "model__num_leaves":    [15, 31, 63],
        "model__learning_rate": [0.05, 0.1],
    },
    "LightGBM_balanced": {
        "model__n_estimators":  [100, 200],
        "model__num_leaves":    [15, 31, 63],
        "model__learning_rate": [0.05, 0.1],
    },
    "LightGBM_SMOTE": {
        "model__n_estimators":  [100, 200],
        "model__num_leaves":    [15, 31, 63],
        "model__learning_rate": [0.05, 0.1],
    },
    "LogisticRegression_default": {
        "model__C": [0.01, 0.1, 1.0],
    },
    "LogisticRegression_balanced": {
        "model__C": [0.01, 0.1, 1.0],
    },
    "LogisticRegression_SMOTE": {
        "model__C": [0.01, 0.1, 1.0],
    },
}


def _rf_base(class_weight=None):
    cfg = get_config('random_forest')
    cfg['class_weight'] = class_weight
    return RandomForestClassifier(**cfg)


def _xgb_base(scale_pos_weight=None):
    cfg = get_config('xgboost')
    if scale_pos_weight is not None:
        cfg['scale_pos_weight'] = scale_pos_weight
    return XGBClassifier(**cfg)


def _lgbm_base(class_weight=None):
    cfg = get_config('lightgbm')
    cfg['class_weight'] = class_weight
    return LGBMClassifier(**cfg)


def _lr_base(class_weight=None):
    cfg = get_config('logistic_regression')
    cfg['class_weight'] = class_weight
    return LogisticRegression(**cfg)


def build_classical_models(pos_neg_ratio: float = 1.0) -> dict:
    """Build all classical models as Pipelines with internal MinMaxScaler.
    Returns dict of model_name -> Pipeline. Expects unscaled input.
    MinMax [0,1] so the classical model input space matches the deep models
    and every attack budget is a min-max L-inf distance (trees are invariant
    to monotone scaling; KNN/MLP see the [0,1] representation)."""

    def std_pipe(estimator):
        return Pipeline([
            ("scaler", MinMaxScaler()),
            ("model",  estimator),
        ])

    def smote_pipe(estimator):
        return ImbPipeline([
            ("scaler", MinMaxScaler()),
            ("smote",  SMOTE(random_state=42, k_neighbors=5)),
            ("model",  estimator),
        ])

    models = {}

    # RandomForest
    models["RandomForest_default"] = std_pipe(_rf_base(class_weight=None))
    models["RandomForest_balanced"] = std_pipe(
        _rf_base(class_weight='balanced'))
    models["RandomForest_SMOTE"] = smote_pipe(_rf_base(class_weight=None))

    # XGBoost
    models["XGBoost_default"] = std_pipe(_xgb_base())
    models["XGBoost_balanced"] = std_pipe(
        _xgb_base(scale_pos_weight=pos_neg_ratio))
    models["XGBoost_SMOTE"] = smote_pipe(_xgb_base())

    # LightGBM
    models["LightGBM_default"] = std_pipe(_lgbm_base(class_weight=None))
    models["LightGBM_balanced"] = std_pipe(_lgbm_base(class_weight='balanced'))
    models["LightGBM_SMOTE"] = smote_pipe(_lgbm_base(class_weight=None))

    # LogisticRegression
    models["LogisticRegression_default"] = std_pipe(
        _lr_base(class_weight=None))
    models["LogisticRegression_balanced"] = std_pipe(
        _lr_base(class_weight='balanced'))
    models["LogisticRegression_SMOTE"] = smote_pipe(
        _lr_base(class_weight=None))

    # Single-variant models
    models["GradientBoosting"] = std_pipe(
        GradientBoostingClassifier(**get_config('gradient_boosting')))
    models["KNN"] = std_pipe(KNeighborsClassifier(**get_config('knn')))
    models["DecisionTree"] = std_pipe(
        DecisionTreeClassifier(**get_config('decision_tree')))
    models["MLP"] = std_pipe(
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
    )

    return models


def tune_model(name: str, model, X_train, y_train, cv):
    """Run GridSearchCV if a param grid exists for this model."""
    if name not in HYPERPARAM_GRIDS:
        return model, None

    print(f"  ⟳  Running GridSearchCV for {name}...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=HYPERPARAM_GRIDS[name],
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    print(f"  ✓  Best params: {grid_search.best_params_}")
    print(f"  ✓  Best CV F1:  {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_, grid_search.best_params_


def calibrate_model(model, X_val, y_val):
    """Wrap fitted model in CalibratedClassifierCV (isotonic, cv=5)."""
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method='isotonic',
        cv=5,
    )
    calibrated.fit(X_val, y_val)
    return calibrated


def calibration_gap(model, X, y, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE). Lower is better."""
    if not hasattr(model, 'predict_proba'):
        return float('nan')
    prob_true, prob_pred = calibration_curve(
        y, model.predict_proba(X)[:, 1],
        n_bins=n_bins, strategy='uniform'
    )
    bin_sizes = np.histogram(
        model.predict_proba(X)[:, 1], bins=n_bins, range=(0, 1)
    )[0]
    weights = bin_sizes / bin_sizes.sum()
    return float(np.sum(weights * np.abs(prob_true - prob_pred)))


def measure_inference_time(model, X, n_repeats: int = 3) -> float:
    """Median per-sample inference time in microseconds."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        model.predict(X)
        times.append((time.perf_counter() - t0) / len(X) * 1e6)
    return float(np.median(times))


def calculate_pr_auc(model, X, y):
    """PR-AUC; returns None if model has no predict_proba."""
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[:, 1]
        precision, recall, _ = precision_recall_curve(y, proba)
        return float(auc(recall, precision))
    return None


def calculate_brier_score(model, X, y) -> float:
    """Brier score (MSE of predicted probabilities vs true labels)."""
    if not hasattr(model, 'predict_proba'):
        return float('nan')
    return float(brier_score_loss(y, model.predict_proba(X)[:, 1]))


def threshold_analysis(
    model,
    X,
    y,
    min_recall: float = 0.95,
    n_thresholds: int = 181,
) -> dict:
    """Sweep thresholds to find max-F1 subject to recall >= min_recall."""
    if not hasattr(model, 'predict_proba'):
        return None

    proba = model.predict_proba(X)[:, 1]
    thresholds = np.linspace(0.05, 0.95, n_thresholds)

    records = []
    for t in thresholds:
        preds = (proba >= t).astype(int)
        tp = int(np.sum((preds == 1) & (y == 1)))
        fp = int(np.sum((preds == 1) & (y == 0)))
        fn = int(np.sum((preds == 0) & (y == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_val = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        records.append({'threshold': float(t),
                        'precision': prec,
                        'recall':    rec,
                        'f1':        f1_val})

    df = pd.DataFrame(records)

    feasible = df[df['recall'] >= min_recall]
    constraint_feasible = not feasible.empty

    if constraint_feasible:
        best_row = feasible.loc[feasible['f1'].idxmax()]
    else:
        best_row = df.loc[df['recall'].idxmax()]

    default_row = df.iloc[(df['threshold'] - 0.5).abs().argsort()[:1]].iloc[0]

    return {
        'threshold_df':          df,
        'optimal_threshold':     float(best_row['threshold']),
        'optimal_precision':     float(best_row['precision']),
        'optimal_recall':        float(best_row['recall']),
        'optimal_f1':            float(best_row['f1']),
        'default_precision':     float(default_row['precision']),
        'default_recall':        float(default_row['recall']),
        'default_f1':            float(default_row['f1']),
        'min_recall_constraint': min_recall,
        'constraint_feasible':   constraint_feasible,
    }


def plot_threshold_analysis(threshold_results: dict, model_name: str,
                            save_dir: str = "baseline"):
    """Plot precision/recall/F1 vs threshold with optimal annotation."""
    if threshold_results is None:
        return

    df = threshold_results['threshold_df']
    opt_t = threshold_results['optimal_threshold']
    opt_r = threshold_results['optimal_recall']
    opt_p = threshold_results['optimal_precision']
    opt_f1 = threshold_results['optimal_f1']
    min_rec = threshold_results['min_recall_constraint']

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df['threshold'], df['precision'], label='Precision',
            color='steelblue',  lw=2)
    ax.plot(df['threshold'], df['recall'],    label='Recall',
            color='darkorange', lw=2)
    ax.plot(df['threshold'], df['f1'],        label='F1',
            color='seagreen',   lw=2)

    ax.axhline(min_rec, color='darkorange', lw=1, ls='--', alpha=0.4)
    ax.axhspan(0, min_rec, alpha=0.04, color='darkorange',
               label=f'Recall < {min_rec} (forbidden)')
    ax.axvline(0.5, color='grey', lw=1.2, ls='--',
               label='Default threshold (0.5)')
    ax.axvline(opt_t, color='red', lw=1.5, ls='--',
               label=(f'Optimal threshold ({opt_t:.2f})\n'
                      f'P={opt_p:.3f} R={opt_r:.3f} F1={opt_f1:.3f}'))

    ax.set_xlabel('Decision Threshold', fontsize=12)
    ax.set_ylabel('Score',              fontsize=12)
    ax.set_title(f'Threshold Analysis — {model_name}', fontsize=13)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    figures_dir = TABLES_DIR.parent / 'figures' / save_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / f"threshold_{model_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Threshold plot saved: {out_path}")


def train_baseline_models(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    pos_neg_ratio: float = 1.0,
    save_models: bool = True,
) -> tuple:
    """Train all baseline models with CV, tuning, calibration, and threshold analysis.
    Input arrays are feature-engineered but NOT scaled (Pipelines scale internally).
    Saves optimal_thresholds.csv as a per-model diagnostic (each stage after
    this one computes its own operating-point threshold independently)."""
    print("\n" + "=" * 70)
    print("BASELINE MODEL TRAINING")
    print("=" * 70)

    models = build_classical_models(pos_neg_ratio=pos_neg_ratio)
    results = []
    trained = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n{'='*70}")
        print(f"Training: {name}")
        print(f"{'='*70}")

        start_time = time.time()

        # Step 1: 5-fold CV (untuned model for fair comparison)
        print("  Running 5-fold CV...")
        cv_results = cross_validate(
            model, X_train, y_train,
            cv=cv,
            scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
            n_jobs=-1,
            return_train_score=False,
        )
        cv_f1_mean = float(np.mean(cv_results['test_f1']))
        cv_f1_std = float(np.std(cv_results['test_f1']))
        cv_auc_mean = float(np.mean(cv_results['test_roc_auc']))
        cv_auc_std = float(np.std(cv_results['test_roc_auc']))
        print(f"  ✓ CV F1:  {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")
        print(f"  ✓ CV AUC: {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")

        # Step 2: Hyperparameter tuning
        model, best_params = tune_model(name, model, X_train, y_train, cv)
        if best_params is None:
            model.fit(X_train, y_train)

        training_time = time.time() - start_time
        print(f"  ✓ Training complete in {training_time:.2f}s")

        # Step 3: Calibration on validation set
        print("  Calibrating probabilities on validation set...")
        try:
            calibrated_model = calibrate_model(model, X_val, y_val)
            ece_before = calibration_gap(model,            X_val, y_val)
            ece_after = calibration_gap(calibrated_model, X_val, y_val)
            print(f"  ✓ ECE before: {ece_before:.4f}  after: {ece_after:.4f}")
            calibration_applied = True
        except Exception as exc:
            print(f"  ⚠ Calibration skipped ({exc})")
            calibrated_model = model
            ece_before = float('nan')
            ece_after = float('nan')
            calibration_applied = False

        # Step 4: Validation metrics
        val_metrics = evaluate_model(calibrated_model, X_val, y_val,
                                     model_name=f"{name}_val")
        print(f"\n  Validation (post-calibration): "
              f"Acc={val_metrics['accuracy']:.4f}  "
              f"F1={val_metrics['f1']:.4f}  "
              f"AUC={val_metrics['auc_roc']:.4f}")

        # Step 5: Test metrics
        test_metrics = evaluate_model(calibrated_model, X_test, y_test,
                                      model_name=name)
        train_metrics = evaluate_model(calibrated_model, X_train, y_train,
                                       model_name=f"{name}_train")
        inference_us = measure_inference_time(calibrated_model, X_test)
        pr_auc = calculate_pr_auc(calibrated_model, X_test, y_test)
        brier = calculate_brier_score(calibrated_model, X_test, y_test)

        # Step 6: Threshold analysis
        t_results = threshold_analysis(
            calibrated_model, X_test, y_test, min_recall=0.95)

        # Imbalance strategy tag
        if "_SMOTE" in name:
            imbalance_strategy = "SMOTE"
        elif "_balanced" in name:
            imbalance_strategy = "balanced"
        else:
            imbalance_strategy = "default"

        test_metrics.update({
            'model_name':           name,
            'imbalance_strategy':   imbalance_strategy,
            'hyperparameter_tuned': best_params is not None,
            'best_params':          str(best_params) if best_params else "N/A",
            'calibration_applied':  calibration_applied,
            'ece_before':           ece_before,
            'ece_after':            ece_after,
            'training_time':        training_time,
            'cv_f1_mean':           cv_f1_mean,
            'cv_f1_std':            cv_f1_std,
            'cv_auc_mean':          cv_auc_mean,
            'cv_auc_std':           cv_auc_std,
            'val_accuracy':         val_metrics['accuracy'],
            'val_f1':               val_metrics['f1'],
            'val_auc_roc':          val_metrics['auc_roc'],
            'train_f1':             train_metrics['f1'],
            'overfitting_gap':      train_metrics['f1'] - test_metrics['f1'],
            'inference_us':         inference_us,
            'pr_auc':               pr_auc,
            'brier_score':          brier,
            'thresh_optimal':       t_results['optimal_threshold'] if t_results else float('nan'),
            'thresh_opt_precision': t_results['optimal_precision'] if t_results else float('nan'),
            'thresh_opt_recall':    t_results['optimal_recall'] if t_results else float('nan'),
            'thresh_opt_f1':        t_results['optimal_f1'] if t_results else float('nan'),
            'thresh_default_f1':    t_results['default_f1'] if t_results else float('nan'),
            'thresh_f1_gain':       (
                t_results['optimal_f1'] - t_results['default_f1']
                if t_results else float('nan')
            ),
        })

        results.append(test_metrics)
        trained[name] = calibrated_model

        print(f"\n  Test Performance:")
        print(f"    Accuracy:       {test_metrics['accuracy']:.4f}")
        print(f"    Precision:      {test_metrics['precision']:.4f}")
        print(f"    Recall:         {test_metrics['recall']:.4f}")
        print(f"    F1-Score:       {test_metrics['f1']:.4f}")
        print(f"    AUC-ROC:        {test_metrics['auc_roc']:.4f}")
        if pr_auc:
            print(f"    PR-AUC:         {pr_auc:.4f}")
        print(f"    Brier score:    {brier:.4f}")
        print(f"    ECE (after):    {ece_after:.4f}")
        print(f"    Train F1:       {train_metrics['f1']:.4f}")
        print(f"    Overfit gap:    {test_metrics['overfitting_gap']:.4f}")
        print(f"    Inference time: {inference_us:.2f} µs/sample")
        if t_results:
            feas = "✓" if t_results['constraint_feasible'] else "⚠ infeasible"
            print(
                f"\n  Threshold Analysis (recall >= 0.95 constraint {feas}):")
            print(f"    Default  t=0.50 → "
                  f"P={t_results['default_precision']:.4f}  "
                  f"R={t_results['default_recall']:.4f}  "
                  f"F1={t_results['default_f1']:.4f}")
            print(f"    Optimal  t={t_results['optimal_threshold']:.2f} → "
                  f"P={t_results['optimal_precision']:.4f}  "
                  f"R={t_results['optimal_recall']:.4f}  "
                  f"F1={t_results['optimal_f1']:.4f}  "
                  f"(Δ F1 = {test_metrics['thresh_f1_gain']:+.4f})")

        if save_models:
            path = CLASSICAL_MODELS / f"{name}.joblib"
            joblib.dump(calibrated_model, path)
            print(f"  ✓ Model saved: {path}")

        if t_results:
            plot_threshold_analysis(t_results, name, save_dir="baseline")

    results_df = (
        pd.DataFrame(results)
        .sort_values('f1', ascending=False)
        .reset_index(drop=True)
    )

    # Save per-model optimal-threshold diagnostic (informational; downstream
    # stages compute their own recall-0.95 operating point independently).
    thresh_rows = []
    for r in results:
        t_opt = r.get('thresh_optimal', float('nan'))
        if not np.isnan(t_opt):
            thresh_rows.append({
                'model':             r['model_name'],
                'optimal_threshold': t_opt,
                'optimal_recall':    r.get('thresh_opt_recall', float('nan')),
                'optimal_precision': r.get('thresh_opt_precision', float('nan')),
                'optimal_f1':        r.get('thresh_opt_f1', float('nan')),
                'default_f1':        r.get('thresh_default_f1', float('nan')),
                'f1_gain':           r.get('thresh_f1_gain', float('nan')),
            })

    if thresh_rows:
        thresh_df = pd.DataFrame(thresh_rows)
        thresh_path = TABLES_DIR / 'optimal_thresholds.csv'
        thresh_df.to_csv(thresh_path, index=False)
        print(f"\n✓ Optimal thresholds saved: {thresh_path}")
    else:
        print("\n⚠ No threshold data to save — all models lacked predict_proba?")

    return results_df, trained


def main():
    print("Setting up directories...")
    create_directories()

    # Step 1: Data loading
    print("\n" + "=" * 70)
    print("STEP 1: DATA LOADING")
    print("=" * 70)
    # Leakage-free block-temporal split of the 9 independent FGI observables
    # (see data.loader.load_track_splits: contiguous temporal blocks per
    # scenario/PRN/segment, with purge gaps — no random shuffling, so adjacent
    # near-identical epochs never straddle train/test). Classical Pipelines scale
    # internally, so they take the UNSCALED X; the returned MinMaxScaler is
    # saved for the DL baseline (02).
    (X_train, X_val, X_test, y_train, y_val, y_test,
     feature_names, scaler) = load_track_splits(verbose=True)
    numeric_cols = feature_names
    print(f"✓ Features: {len(feature_names)}  "
          f"Train/Val/Test: {len(X_train):,}/{len(X_val):,}/{len(X_test):,}")
    print(f"✓ Train label dist: {np.bincount(y_train)}")

    counts = np.bincount(y_train)
    pos_neg_ratio = float(counts[0] / counts[1]) if len(counts) > 1 and counts[1] > 0 else 1.0
    print(f"✓ Negative/Positive ratio (train): {pos_neg_ratio:.2f}")

    assert X_test.shape[0] == len(
        y_test),           "Test sample count mismatch"
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], \
        "Feature dimension mismatch"

    print(f"\n✓ Final feature count: {X_train.shape[1]}")
    print(
        f"✓ Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # Save artifacts for 02_deep_learning_baseline.py
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_FEATURE_NAMES_PATH, 'wb') as f:
        pickle.dump(numeric_cols, f)
    print(f"✓ Raw feature names saved: {RAW_FEATURE_NAMES_PATH}")

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ MinMaxScaler saved: {SCALER_PATH}")

    # Step 5: Model training
    print("\n" + "=" * 70)
    print("STEP 5: MODEL TRAINING")
    print("=" * 70)
    results_df, trained_models = train_baseline_models(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        pos_neg_ratio=pos_neg_ratio,
        save_models=True,
    )

    # Step 6: Save & summarise
    results_path = TABLES_DIR / "baseline_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved: {results_path}")

    print("\n" + "=" * 70)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 70)
    summary_cols = [
        'model_name', 'imbalance_strategy', 'hyperparameter_tuned',
        'accuracy', 'precision', 'recall', 'f1',
        'auc_roc', 'pr_auc', 'brier_score', 'ece_after',
        'cv_f1_mean', 'cv_f1_std', 'inference_us', 'training_time',
    ]
    print(results_df[summary_cols].to_string(index=False))

    print("\n" + "=" * 70)
    print("IMBALANCE STRATEGY COMPARISON  (RandomForest example)")
    print("=" * 70)
    rf_rows = results_df[results_df['model_name'].str.startswith(
        'RandomForest')]
    if not rf_rows.empty:
        print(rf_rows[['model_name', 'imbalance_strategy',
                       'precision', 'recall', 'f1',
                       'pr_auc', 'ece_after']].to_string(index=False))

    comparison = compare_models(results_df.to_dict('records'))
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (sorted by F1)")
    print("=" * 70)
    print(comparison.to_string(index=False))

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS SUMMARY  (recall >= 0.95 constraint)")
    print("=" * 70)
    thresh_cols = [
        'model_name', 'thresh_optimal',
        'thresh_opt_precision', 'thresh_opt_recall',
        'thresh_opt_f1', 'thresh_default_f1', 'thresh_f1_gain',
    ]
    thresh_summary = (
        results_df[thresh_cols]
        .dropna(subset=['thresh_f1_gain'])
        .sort_values('thresh_f1_gain', ascending=False)
    )
    print(thresh_summary.to_string(index=False))

    meaningful = thresh_summary[thresh_summary['thresh_f1_gain'] > 0.01]
    if not meaningful.empty:
        print(
            f"\n  ⟹  {len(meaningful)} model(s) gain >0.01 F1 from threshold tuning:")
        for _, row in meaningful.iterrows():
            print(f"     {row['model_name']:35s}  "
                  f"t={row['thresh_optimal']:.2f}  "
                  f"Δ F1={row['thresh_f1_gain']:+.4f}")

    # Step 7: Visualisations
    print("\n" + "=" * 70)
    print("STEP 7: GENERATING VISUALISATIONS")
    print("=" * 70)
    plot_roc_curves(trained_models, X_test, y_test,
                    save_path="baseline/roc_curves.png")
    plot_confusion_matrices(trained_models, X_test, y_test,
                            save_dir="baseline")
    plot_model_comparison(
        results_df, save_path="baseline/model_comparison.png")

    print("\n" + "=" * 70)
    print("✓ BASELINE EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nBest model:  {results_df.iloc[0]['model_name']} "
          f"(F1={results_df.iloc[0]['f1']:.4f})")
    print(f"Worst model: {results_df.iloc[-1]['model_name']} "
          f"(F1={results_df.iloc[-1]['f1']:.4f})")
    print(f"\nResults  → {TABLES_DIR}")
    print(f"Figures  → {TABLES_DIR.parent / 'figures' / 'baseline'}")


if __name__ == "__main__":
    main()
