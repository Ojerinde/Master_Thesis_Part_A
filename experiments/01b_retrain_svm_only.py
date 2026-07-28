"""
Standalone SVM Retrain -- retrains and hyperparameter-tunes ONLY the
RBF-kernel SVM classical model. Every other detector (16 other classical
variants, 6 deep learning models) is reused as-is from a prior full run.

Why this script exists: SVM was added to build_classical_models() alongside
the other 17 classical variants, but shipped with no entry in
HYPERPARAM_GRIDS, so it trained once at its config default (C=1.0,
gamma='scale') while every other classical model went through 5-fold
GridSearchCV. That gap is now fixed in 01_classical_baseline.py's
HYPERPARAM_GRIDS. Rerunning the full 01_classical_baseline.py (~4-5h --
GridSearchCV over the other 17 variants that were already correctly tuned)
plus 02_deep_learning_baseline.py (~1-2h -- 6 architectures x 3 seeds, none
of which depend on SVM at all) just to fix one model's tuning is pure waste.

What this script does:
  1. Reproduces the IDENTICAL deterministic block-temporal train/val/test
     split (data.loader.load_track_splits has no shuffling / random_state --
     every independent call, in every stage/process of this pipeline,
     returns the same partition; this is the same assumption every
     downstream stage already relies on when it re-derives the split itself).
  2. Builds the full 18-model classical zoo via build_classical_models() (so
     the SVM Pipeline is constructed byte-identically to the full run) and
     extracts only the "SVM" entry.
  3. Runs that one model through the exact same CV / tune / calibrate /
     evaluate / save path as 01_classical_baseline.py's train_baseline_models
     loop -- by loading that module directly with importlib (its filename
     starts with a digit, so `from experiments.01_classical_baseline import
     ...` is not valid Python), reusing its functions byte-for-byte instead
     of duplicating ~300 lines of evaluation logic that must stay in sync.
  4. Merges the new SVM row into baseline_results.csv / optimal_thresholds.csv
     if a prior run's copies are present (keeps the other 17 models' rows),
     otherwise writes a single-row table.

Prerequisite: results/models/classical/*.joblib (the other 17 files) and
results/models/deep_learning/*.pt must already be in place from a prior full
run (e.g. downloaded from the bn_tok Kaggle run and unzipped into place) --
this script does not touch them. See KAGGLE_RUNBOOK.md, "Retrain SVM only."

Usage:
    PYTHONPATH=. python experiments/01b_retrain_svm_only.py
"""
from pathlib import Path
import sys
import time
import importlib.util
import numpy as np
import pandas as pd
import joblib

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.model_selection import StratifiedKFold, cross_validate  # noqa: E402
from data.loader import load_track_splits                              # noqa: E402
from config.paths import create_directories, CLASSICAL_MODELS, TABLES_DIR  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "baseline01", Path(__file__).parent / "01_classical_baseline.py")
baseline01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline01)


def _merge_csv_row(path: Path, row: dict, key_col: str):
    """Load path if it exists, drop any existing row where key_col ==
    row[key_col], append row, save. Writes a fresh single-row file if the
    prior table isn't present (e.g. a from-scratch environment)."""
    new_row = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing[key_col] != row[key_col]]
        merged = pd.concat([existing, new_row], ignore_index=True)
        print(f"  ✓ Merged into existing {path.name} "
              f"({len(existing)} prior rows + 1 new SVM row)")
    else:
        merged = new_row
        print(f"  ⚠ No prior {path.name} found -- writing SVM-only row "
              f"(prior run's table was not present in this environment)")
    merged.to_csv(path, index=False)
    return merged


def main():
    print("=" * 70)
    print("STANDALONE SVM RETRAIN (tuned) -- other 13 detectors reused as-is")
    print("=" * 70)
    create_directories()

    (X_train, X_val, X_test, y_train, y_val, y_test,
     feature_names, _scaler) = load_track_splits(verbose=True)
    print(f"✓ Features: {len(feature_names)}  "
          f"Train/Val/Test: {len(X_train):,}/{len(X_val):,}/{len(X_test):,}")

    counts = np.bincount(y_train)
    pos_neg_ratio = float(counts[0] / counts[1]) if len(counts) > 1 and counts[1] > 0 else 1.0

    # Built via the same function the full run uses, so this Pipeline
    # (MinMaxScaler + _svm_rbf_base()) is byte-identical to the original --
    # only the "SVM" entry is used, the other 17 are discarded.
    name = "SVM"
    model = baseline01.build_classical_models(pos_neg_ratio=pos_neg_ratio)[name]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n{'='*70}\nTraining: {name}\n{'='*70}")
    start_time = time.time()

    print("  Running 5-fold CV...")
    cv_results = cross_validate(
        model, X_train, y_train, cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        n_jobs=-1, return_train_score=False,
    )
    cv_f1_mean = float(np.mean(cv_results['test_f1']))
    cv_f1_std = float(np.std(cv_results['test_f1']))
    cv_auc_mean = float(np.mean(cv_results['test_roc_auc']))
    cv_auc_std = float(np.std(cv_results['test_roc_auc']))
    print(f"  ✓ CV F1:  {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")
    print(f"  ✓ CV AUC: {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")

    # HYPERPARAM_GRIDS['SVM'] (model__C, model__gamma) only matches the cuML
    # backend's Pipeline shape (model step = cuml.svm.SVC directly). The CPU
    # RFF-approximation fallback nests a differently-parameterised
    # sub-pipeline (model__rff__gamma, model__sgd__alpha) and raises
    # ValueError on this grid -- same guard as train_baseline_models()'s call
    # site in 01_classical_baseline.py: substitute a name that won't match
    # HYPERPARAM_GRIDS so tune_model() short-circuits to "no grid, use as-is"
    # instead of crashing.
    tune_name = name if baseline01._CumlSVC is not None else \
        'SVM (CPU fallback, tuning grid does not apply)'
    # n_jobs=1 (GPU-safe) + fallback_on_error=True (this cuML/GridSearchCV
    # combination is genuinely untested on this hardware -- see tune_model's
    # docstring in 01_classical_baseline.py) when cuML is the active backend.
    tune_fallback = baseline01._CumlSVC is not None
    model, best_params = baseline01.tune_model(
        tune_name, model, X_train, y_train, cv, n_jobs=1,
        fallback_on_error=tune_fallback)
    if best_params is None:
        model.fit(X_train, y_train)

    training_time = time.time() - start_time
    print(f"  ✓ Training complete in {training_time:.2f}s")

    print("  Calibrating probabilities on validation set...")
    try:
        calibrated_model = baseline01.calibrate_model(model, X_val, y_val)
        ece_before = baseline01.calibration_gap(model, X_val, y_val)
        ece_after = baseline01.calibration_gap(calibrated_model, X_val, y_val)
        print(f"  ✓ ECE before: {ece_before:.4f}  after: {ece_after:.4f}")
        calibration_applied = True
    except Exception as exc:
        print(f"  ⚠ Calibration skipped ({exc})")
        calibrated_model = model
        ece_before = float('nan')
        ece_after = float('nan')
        calibration_applied = False

    val_metrics = baseline01.evaluate_model(calibrated_model, X_val, y_val,
                                            model_name=f"{name}_val")
    print(f"\n  Validation (post-calibration): "
          f"Acc={val_metrics['accuracy']:.4f}  "
          f"F1={val_metrics['f1']:.4f}  AUC={val_metrics['auc_roc']:.4f}")

    test_metrics = baseline01.evaluate_model(calibrated_model, X_test, y_test,
                                             model_name=name)
    train_metrics = baseline01.evaluate_model(calibrated_model, X_train, y_train,
                                              model_name=f"{name}_train")
    inference_us = baseline01.measure_inference_time(calibrated_model, X_test)
    pr_auc = baseline01.calculate_pr_auc(calibrated_model, X_test, y_test)
    brier = baseline01.calculate_brier_score(calibrated_model, X_test, y_test)
    t_results = baseline01.threshold_analysis(
        calibrated_model, X_test, y_test, min_recall=0.95)

    test_metrics.update({
        'model_name':           name,
        'imbalance_strategy':   'default',
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
    if best_params:
        print(f"    Tuned params:   {best_params}")
    else:
        print(f"    Tuned params:   NONE -- ran at config default "
              f"(tuning skipped or fell back; check the [WARN] above)")

    path = CLASSICAL_MODELS / f"{name}.joblib"
    joblib.dump(calibrated_model, path)
    print(f"\n  ✓ Model saved: {path}")
    print(f"  This OVERWRITES the SVM.joblib from any prior run. The other "
          f"13 detectors' model files are untouched.")

    if t_results:
        baseline01.plot_threshold_analysis(t_results, name, save_dir="baseline")

    print("\n" + "=" * 70)
    print("UPDATING DIAGNOSTIC TABLES")
    print("=" * 70)
    _merge_csv_row(TABLES_DIR / "baseline_results.csv", test_metrics, "model_name")

    if t_results:
        thresh_row = {
            'model':             name,
            'optimal_threshold': test_metrics['thresh_optimal'],
            'optimal_recall':    test_metrics['thresh_opt_recall'],
            'optimal_precision': test_metrics['thresh_opt_precision'],
            'optimal_f1':        test_metrics['thresh_opt_f1'],
            'default_f1':        test_metrics['thresh_default_f1'],
            'f1_gain':           test_metrics['thresh_f1_gain'],
        }
        _merge_csv_row(TABLES_DIR / "optimal_thresholds.csv", thresh_row, "model")

    print("\n" + "=" * 70)
    print("✓ SVM RETRAIN COMPLETE")
    print("=" * 70)
    print("Next: rerun the downstream stages so every table/figure reflects "
          "the newly-tuned SVM -- see run_pipeline_svm_retrain.py.")


if __name__ == "__main__":
    main()
