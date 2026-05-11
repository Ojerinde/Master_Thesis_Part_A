"""
Adversarial Robustness Evaluation — runs 7 attack types against all trained
GNSS spoofing detectors (classical + DL).

Key design: classical models receive unscaled input (X_test_eng) because their
Pipelines apply StandardScaler internally; DL models receive scaled input
(X_test). Transfer attacks inverse-transform from scaled to unscaled space.

Attack taxonomy:
  FGSM / PGD          — DL white-box gradient attacks
  FGSM/PGD-Transfer   — classical models via CNN-1D surrogate
  DLSA / SNA / TPA    — feature-space attacks (all models)

Usage:
    python -m experiments.03_adversarial_evaluation [--quick] [--epsilons 0.05 0.10]
"""

from models.deep_learning import (
    CNN1DModel, LSTMModel, BiLSTMModel,
    CNNLSTMModel, TransformerModel, TCNModel,
)
from utils.gnss_constraints import GNSSConstraintEnforcer
from attacks.gnss_attacks import (
    DataLocationShiftAttack,
    SimilarityNoiseAttack,
    TemporalPatternAttack,
)
from attacks.pgd import PGDAttack
from attacks.fgsm import FGSMAttack
from data.feature_engineering import SafeFeatureEngineer
from data.loader import load_texbat
from config.model_configs import get_config
from config.paths import create_directories, CLASSICAL_MODELS, DL_MODELS, TABLES_DIR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
import sys
import time
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})


# --- Configuration ---

EPSILONS = [0.05, 0.10, 0.20]
DLSA_SCALES = [0.05, 0.10, 0.20]
SNA_EPSILONS = [0.05, 0.10, 0.20]
TPA_CONFIGS = [(0.05, 0.03), (0.10, 0.05), (0.20, 0.10)]

INFER_REPEATS = 3

DL_REGISTRY = {
    'CNN-1D':      (CNN1DModel,       'cnn_1d'),
    'LSTM':        (LSTMModel,        'lstm'),
    'BiLSTM':      (BiLSTMModel,      'bilstm'),
    'CNN-LSTM':    (CNNLSTMModel,     'cnn_lstm'),
    'Transformer': (TransformerModel, 'transformer'),
    'TCN':         (TCNModel,         'tcn'),
}

# Classical models included in adversarial evaluation.
# Excluded:
#   LogisticRegression — degenerate baseline (clean recall 0.37)
#   SVM_RBF            — inference 1438 µs/sample, infeasible for real-time GNSS
SELECTED_MODELS = {
    'RandomForest':     'RandomForest_SMOTE',
    'XGBoost':          'XGBoost_default',
    'LightGBM':         'LightGBM_default',
    'GradientBoosting': 'GradientBoosting',
    'KNN':              'KNN',
    'MLP':              'MLP',
    'DecisionTree':     'DecisionTree',
}


# --- Threshold loading ---

def load_optimal_thresholds() -> dict:
    """
    Load per-model optimal thresholds saved by 01_classical_baseline.py.

    Returns dict: model_stem -> optimal_threshold (float).
    Falls back to 0.5 for any model not in the file, and returns empty
    dict (all models use 0.5) if the file does not exist.
    """
    thresh_path = TABLES_DIR / 'optimal_thresholds.csv'
    if not thresh_path.exists():
        print(f"  ⚠ optimal_thresholds.csv not found at {thresh_path}")
        print("    Run 01_classical_baseline.py first to generate it.")
        print("    Falling back to threshold=0.5 for all models.")
        return {}
    df = pd.read_csv(thresh_path)
    thresholds = dict(zip(df['model'], df['optimal_threshold']))
    print(f"  ✓ Loaded optimal thresholds for {len(thresholds)} models")
    return thresholds


# --- Metrics helpers ---

def _has_proba(model) -> bool:
    """
    Probe whether model supports predict_proba at call time.

    Bug 9 fix: uses float64 dummy. sklearn pipelines can behave unexpectedly
    with float32 inputs in some steps, causing false negatives in the probe.

    sklearn Pipelines expose predict_proba as a descriptor that raises
    AttributeError only at call time — a guarded probe call is the only
    reliable test.
    """
    try:
        n = _get_n_features(model)
        dummy = np.zeros((1, n), dtype=np.float64)
        model.predict_proba(dummy)
        return True
    except (AttributeError, NotImplementedError):
        return False
    except Exception:
        return True   # predict_proba exists; let the real call handle errors


def _get_n_features(model) -> int:
    """Extract expected input feature count from any fitted model."""
    if hasattr(model, 'n_features_in_'):
        return int(model.n_features_in_)
    if hasattr(model, 'named_steps'):
        for step in model.named_steps.values():
            if hasattr(step, 'n_features_in_'):
                return int(step.n_features_in_)
    if hasattr(model, 'input_dim'):
        return int(model.input_dim)
    return 23   # TEXBAT feature count after engineering


def _proba(model, X: np.ndarray) -> np.ndarray:
    """
    Return flat (n,) positive-class probability.
    Falls back to binary predict() for models without predict_proba.
    """
    if _has_proba(model):
        p = model.predict_proba(X)
        if p.ndim == 2:
            p = p[:, 1]
        return p.ravel().astype(np.float64)
    return model.predict(X).astype(np.float64)


def evaluate(model, X: np.ndarray, y: np.ndarray,
             model_name: str = '',
             threshold: float = 0.5) -> dict:
    """
    Full metric set for one (model, dataset) pair.

    threshold: decision threshold applied to predict_proba output.
               Use optimal per-model threshold from optimal_thresholds.csv
               rather than the default 0.5 to avoid recall=1.0 artefact
               on imbalanced data.
    """
    proba = _proba(model, X)
    preds = (proba >= threshold).astype(int)
    try:
        auc = float(roc_auc_score(y, proba))
    except ValueError:
        auc = float('nan')
    return {
        'model':     model_name,
        'threshold': threshold,
        'accuracy':  float(accuracy_score(y, preds)),
        'precision': float(precision_score(y, preds, zero_division=0)),
        'recall':    float(recall_score(y, preds, zero_division=0)),
        'f1':        float(f1_score(y, preds, zero_division=0)),
        'auc_roc':   auc,
    }


def delta_metrics(baseline: dict, adversarial: dict) -> dict:
    return {
        'delta_accuracy':  baseline['accuracy'] - adversarial['accuracy'],
        'delta_f1':        baseline['f1'] - adversarial['f1'],
        'delta_recall':    baseline['recall'] - adversarial['recall'],
        'delta_precision': baseline['precision'] - adversarial['precision'],
    }


def attack_success_rate(model, X_clean: np.ndarray,
                        X_adv: np.ndarray, y: np.ndarray,
                        threshold: float = 0.5) -> float:
    """
    ASR = |{i : f(x_i)=y_i AND f(x̃_i)≠y_i}| / |{i : f(x_i)=y_i}|

    Conditions on prior correctness to avoid inflated rates on imbalanced data.
    Uses per-model optimal threshold (same as evaluate()) for consistency.
    Returns NaN for models without predict_proba.

    Bug 5 fix: dead code after return removed.
    """
    if not _has_proba(model):
        return float('nan')

    proba_clean = _proba(model, X_clean)
    proba_adv = _proba(model, X_adv)
    pred_clean = (proba_clean >= threshold).astype(int)
    pred_adv = (proba_adv >= threshold).astype(int)

    correct_mask = (pred_clean == y)
    n_correct = correct_mask.sum()
    if n_correct == 0:
        return float('nan')

    flipped = (pred_adv[correct_mask] != y[correct_mask]).sum()
    return float(flipped / n_correct)


def measure_inference_latency(model, X: np.ndarray,
                              n_repeats: int = INFER_REPEATS) -> float:
    """Median inference latency in µs per sample."""
    call = model.predict_proba if _has_proba(model) else model.predict
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        call(X)
        times.append((time.perf_counter() - t0) / len(X) * 1e6)
    return float(np.median(times))


# --- Data loading (dual paths: scaled for DL, unscaled for classical) ---

def load_and_preprocess():
    """Load TEXBAT and return dual feature matrices:
    - Scaled (X_train/val/test): for DL models
    - Unscaled (X_train/val/test_eng): for classical Pipelines (avoid double-scaling)
    """
    print("Loading TEXBAT dataset...")
    df, _ = load_texbat(verbose=True, validate=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in ['label', 'Label']:
        if col in numeric_cols:
            numeric_cols.remove(col)
    if 'Label' in df.columns:
        df = df.rename(columns={'Label': 'label'})

    y = df['label'].values.astype(int)
    X_raw = df[numeric_cols].values

    # Identical split to 01_classical_baseline.py (same random_state, fractions)
    X_temp, X_test_raw, y_temp, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp)

    # Feature engineering — fit on train only
    engineer = SafeFeatureEngineer(verbose=False)
    X_train_eng, X_val_eng, feat_names = engineer.fit_transform(
        X_train_raw, X_val_raw, feature_names=numeric_cols)
    X_test_eng = engineer.transform(X_test_raw)

    # Classical path — unscaled (float32 for memory efficiency)
    X_train_eng = X_train_eng.astype(np.float32)
    X_val_eng = X_val_eng.astype(np.float32)
    X_test_eng = X_test_eng.astype(np.float32)

    # DL path — externally scaled
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_eng).astype(np.float32)
    X_val = scaler.transform(X_val_eng).astype(np.float32)
    X_test = scaler.transform(X_test_eng).astype(np.float32)

    print(f"✓ DL  path (scaled):   Train {X_train.shape} | "
          f"Val {X_val.shape} | Test {X_test.shape}")
    print(f"✓ CLF path (unscaled): Train {X_train_eng.shape} | "
          f"Val {X_val_eng.shape} | Test {X_test_eng.shape}")

    return (X_train,     X_val,     X_test,
            X_train_eng, X_val_eng, X_test_eng,
            y_train, y_val, y_test,
            feat_names, scaler)


# --- Preprocessing diagnostic ---

def run_preprocessing_diagnostic(classical_models: dict,
                                 X_test_eng: np.ndarray,
                                 y_test: np.ndarray,
                                 thresholds: dict) -> bool:
    """
    Verify classical models produce sensible predictions on unscaled data.

    Checks:
    1. predicted_spoofing rate should be close to true_spoofing rate (~0.75)
    2. recall at optimal threshold should be < 1.0 (except for genuinely
       perfect models, which is extremely unlikely)

    If any model predicts >98% spoofing even on unscaled data, that indicates
    a more fundamental problem (e.g. wrong model file, corrupt joblib).

    Returns True if all checks pass, False if a critical problem is found.
    """
    print("\n--- Preprocessing diagnostic (classical models, unscaled input) ---")
    problems = []
    for name, model in classical_models.items():
        try:
            thresh = thresholds.get(
                SELECTED_MODELS.get(name, name), 0.5)
            proba = model.predict_proba(X_test_eng.astype(np.float64))[:, 1]
            preds = (proba >= thresh).astype(int)
            spoof_rate = preds.mean()
            true_rate = y_test.mean()
            print(f"  {name:25s}  predicted_spoofing={spoof_rate:.3f}  "
                  f"true_spoofing={true_rate:.3f}  threshold={thresh:.2f}")
            if spoof_rate > 0.98:
                problems.append(name)
        except Exception as e:
            print(f"  {name:25s}  ⚠ Could not evaluate: {e}")

    if problems:
        print(f"\n⚠ CRITICAL: {len(problems)} model(s) predict >98% spoofing "
              f"even on unscaled input:")
        for p in problems:
            print(f"    {p}")
        print("\n  This is NOT a double-scaling issue (unscaled data was passed).")
        print("  Possible causes: wrong model file, corrupt joblib, or model")
        print("  was trained on a different feature set entirely.")
        print("  Halting to prevent reporting of invalid results.\n")
        return False

    print("  ✓ All models produce expected prediction distributions.\n")
    return True


# --- Model loaders ---

def load_classical_models() -> dict:
    """
    Load one representative per algorithm family.

    Returns display_name -> fitted model (Pipeline with internal scaler).
    These models expect UNSCALED feature-engineered input.
    """
    wanted = {v: k for k, v in SELECTED_MODELS.items()}
    models = {}
    skipped = []

    for path in sorted(CLASSICAL_MODELS.glob('*.joblib')):
        stem = path.stem
        if stem not in wanted:
            skipped.append(stem)
            continue
        try:
            models[wanted[stem]] = joblib.load(path)
            print(f"  ✓ {wanted[stem]:20s}  ← {stem}")
        except Exception as e:
            print(f"  ⚠ {wanted[stem]} ({stem}): {e}")

    if skipped:
        print(f"  — Skipped {len(skipped)} variant(s): "
              f"{', '.join(skipped[:8])}"
              f"{'...' if len(skipped) > 8 else ''}")

    return models


def load_dl_models(input_dim: int) -> dict:
    """Load all DL models. These expect SCALED input (no internal scaler)."""
    models = {}
    for name, (ModelClass, config_name) in DL_REGISTRY.items():
        pt_path = DL_MODELS / f"{config_name}.pt"
        if not pt_path.exists():
            print(f"  ⚠ {name}: {pt_path} not found")
            continue
        try:
            config = get_config(config_name)
            config['input_dim'] = input_dim
            m = ModelClass(input_dim=input_dim, config=config)
            m.build_model()
            m.model.load_state_dict(
                torch.load(str(pt_path), map_location='cpu'))
            m.model.eval()
            m.is_trained = True
            models[name] = m
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ⚠ {name}: {e}")
    return models


# --- Gradient attacks (DL models, white-box) ---

def run_gradient_attacks(dl_models: dict,
                         X_test: np.ndarray,
                         y_test: np.ndarray,
                         epsilons: list,
                         feature_names: list,
                         enforcer_dl: GNSSConstraintEnforcer) -> list:
    """
    FGSM and PGD on each DL model (white-box, scaled feature space).

    Bug 4 fix: pre-fitted enforcer passed in — physical constraints enforced.
    Bug 6 fix: generate() called once; result used for both timing and metrics.

    Logs actual L-inf perturbation after enforcer clipping to diagnose
    FGSM non-monotonicity (if actual_linf < epsilon at ε=0.20, the enforcer
    is clipping back, which is the likely cause of non-monotonic performance).
    """
    results = []

    for model_name, model in dl_models.items():
        print(f"\n  {model_name}")
        baseline = evaluate(model, X_test, y_test, model_name, threshold=0.5)
        infer_us = measure_inference_latency(model, X_test)

        for epsilon in epsilons:
            for atk_cls, atk_label, kwargs in [
                (FGSMAttack, 'FGSM', {}),
                (PGDAttack,  'PGD',  {'num_iter': 40, 'random_start': True}),
            ]:
                try:
                    attacker = atk_cls(model, epsilon=epsilon,
                                       gnss_enforcer=enforcer_dl, **kwargs)

                    # Bug 6 fix: single generate() call, timed
                    t0 = time.perf_counter()
                    X_adv = attacker.generate(X_test, y_test)
                    atk_gen_us = (time.perf_counter() - t0) / len(X_test) * 1e6

                    # Diagnose enforcer clipping
                    actual_linf = float(np.max(np.abs(X_adv - X_test)))
                    if actual_linf < epsilon * 0.9:
                        print(f"    ⚠ Enforcer clipping: requested ε={epsilon:.2f}  "
                              f"actual L-inf={actual_linf:.4f}  "
                              f"({actual_linf/epsilon*100:.0f}% of budget used)")
                    else:
                        print(f"    Requested ε={epsilon:.2f}  "
                              f"Actual L-inf={actual_linf:.4f}")

                    adv = evaluate(model, X_adv, y_test, model_name,
                                   threshold=0.5)
                    d = delta_metrics(baseline, adv)
                    asr = attack_success_rate(model, X_test, X_adv, y_test,
                                              threshold=0.5)

                    print(f"    {atk_label} ε={epsilon:.2f}  "
                          f"F1 {baseline['f1']:.4f}→{adv['f1']:.4f}  "
                          f"ΔF1={d['delta_f1']:+.4f}  "
                          f"Recall={adv['recall']:.4f}  "
                          f"ASR={asr:.3f}  "
                          f"gen={atk_gen_us:.1f}µs/sample")

                    results.append({
                        'model':      model_name,
                        'model_type': 'deep_learning',
                        'attack':     atk_label,
                        'epsilon':    epsilon,
                        'threshold':  0.5,
                        'actual_linf': actual_linf,
                        'baseline_accuracy':  baseline['accuracy'],
                        'baseline_f1':        baseline['f1'],
                        'baseline_recall':    baseline['recall'],
                        'baseline_precision': baseline['precision'],
                        'baseline_auc_roc':   baseline['auc_roc'],
                        'adv_accuracy':  adv['accuracy'],
                        'adv_f1':        adv['f1'],
                        'adv_recall':    adv['recall'],
                        'adv_precision': adv['precision'],
                        'adv_auc_roc':   adv['auc_roc'],
                        **d,
                        'attack_success_rate': asr,
                        'detector_infer_us':   infer_us,
                        'attack_gen_us':        atk_gen_us,
                        'latency_ratio':        atk_gen_us / max(infer_us, 1e-9),
                    })

                except Exception as e:
                    print(f"    {atk_label} ε={epsilon:.2f}: FAILED — {e}")
                    import traceback
                    traceback.print_exc()

    return results


# --- Transfer attacks (classical models via CNN-1D surrogate) ---

def run_transfer_attacks(dl_models: dict,
                         classical_models: dict,
                         X_test: np.ndarray,
                         X_test_eng: np.ndarray,
                         y_test: np.ndarray,
                         epsilons: list,
                         feature_names: list,
                         enforcer_dl: GNSSConstraintEnforcer,
                         scaler: StandardScaler,
                         thresholds: dict) -> list:
    """
    Craft adversarial examples on CNN-1D surrogate (scaled DL space),
    then evaluate on classical models (unscaled space).

    Data path:
    1. Generate adversarial examples on surrogate using X_test (scaled).
    2. Inverse-transform X_adv back to unscaled feature-engineered space
       using scaler.inverse_transform().
    3. Evaluate classical models on X_adv_eng (unscaled adversarial).
    4. Classical models' Pipelines then re-apply their internal StandardScaler
       — this is correct; the perturbations are preserved in the features.

    Bug 4 fix: pre-fitted enforcer_dl used for surrogate attacks.
    Bug 6 fix: generate() called once per (epsilon, attack); result reused.
    """
    results = []

    surrogate_name = 'CNN-1D' if 'CNN-1D' in dl_models else next(
        iter(dl_models), None)
    if surrogate_name is None:
        print("  ⚠ No DL surrogate available for transfer attacks.")
        return results

    surrogate = dl_models[surrogate_name]
    print(f"\n  Surrogate: {surrogate_name}")

    for epsilon in epsilons:
        # Generate adversarial examples in scaled DL space (once per epsilon)
        fgsm = FGSMAttack(surrogate, epsilon=epsilon,
                          gnss_enforcer=enforcer_dl)
        pgd_att = PGDAttack(surrogate, epsilon=epsilon, num_iter=40,
                            random_start=True, gnss_enforcer=enforcer_dl)

        X_adv_fgsm_scaled = fgsm.generate(X_test, y_test)
        X_adv_pgd_scaled = pgd_att.generate(X_test, y_test)

        # Inverse-transform to unscaled space for classical evaluation
        X_adv_fgsm_eng = scaler.inverse_transform(
            X_adv_fgsm_scaled).astype(np.float32)
        X_adv_pgd_eng = scaler.inverse_transform(
            X_adv_pgd_scaled).astype(np.float32)

        for model_name, model in classical_models.items():
            stem = SELECTED_MODELS.get(model_name, model_name)
            thresh = thresholds.get(stem, 0.5)
            baseline = evaluate(model, X_test_eng, y_test, model_name,
                                threshold=thresh)
            infer_us = measure_inference_latency(model, X_test_eng)

            for atk_label, X_adv_eng in [
                ('FGSM-Transfer', X_adv_fgsm_eng),
                ('PGD-Transfer',  X_adv_pgd_eng),
            ]:
                try:
                    adv = evaluate(model, X_adv_eng, y_test, model_name,
                                   threshold=thresh)
                    d = delta_metrics(baseline, adv)
                    asr = attack_success_rate(model, X_test_eng, X_adv_eng,
                                              y_test, threshold=thresh)

                    print(f"    {model_name:20s}  {atk_label}  ε={epsilon:.2f}  "
                          f"ΔF1={d['delta_f1']:+.4f}  "
                          f"ΔRecall={d['delta_recall']:+.4f}  "
                          f"ASR={asr:.3f}")

                    results.append({
                        'model':      model_name,
                        'model_type': 'classical',
                        'attack':     atk_label,
                        'epsilon':    epsilon,
                        'surrogate':  surrogate_name,
                        'threshold':  thresh,
                        'baseline_accuracy':  baseline['accuracy'],
                        'baseline_f1':        baseline['f1'],
                        'baseline_recall':    baseline['recall'],
                        'baseline_precision': baseline['precision'],
                        'baseline_auc_roc':   baseline['auc_roc'],
                        'adv_accuracy':  adv['accuracy'],
                        'adv_f1':        adv['f1'],
                        'adv_recall':    adv['recall'],
                        'adv_precision': adv['precision'],
                        'adv_auc_roc':   adv['auc_roc'],
                        **d,
                        'attack_success_rate': asr,
                        'detector_infer_us':   infer_us,
                        'attack_gen_us':        float('nan'),
                        'latency_ratio':        float('nan'),
                    })
                except Exception as e:
                    print(f"    {model_name}  {atk_label} ε={epsilon:.2f}: "
                          f"FAILED — {e}")

    return results


# --- Feature-space attacks (DLSA, SNA, TPA — all models) ---

def run_feature_space_attacks(dl_models: dict,
                              classical_models: dict,
                              X_train: np.ndarray,
                              X_test: np.ndarray,
                              X_train_eng: np.ndarray,
                              X_test_eng: np.ndarray,
                              y_test: np.ndarray,
                              feature_names: list,
                              enforcer_dl: GNSSConstraintEnforcer,
                              enforcer_clf: GNSSConstraintEnforcer,
                              thresholds: dict) -> list:
    """
    DLSA, SNA, TPA on all models.

    Data routing:
      DL models  → X_test (scaled),   enforcer_dl (fitted on scaled train)
      Classical  → X_test_eng (unscaled), enforcer_clf (fitted on unscaled train)

    Bug 3 fix: DLSA uses centroid-based adversarial direction (in gnss_attacks.py).
    Bug 6 fix: generate() called ONCE; result used for timing and evaluation.
    Bug 8 fix: seeded RNGs in SNA and TPA for reproducibility.
    """
    results = []

    # Fit SNA std on both data variants
    sna_base_dl = SimilarityNoiseAttack(gnss_enforcer=enforcer_dl,  seed=42)
    sna_base_dl.fit(X_train)
    sna_base_clf = SimilarityNoiseAttack(gnss_enforcer=enforcer_clf, seed=42)
    sna_base_clf.fit(X_train_eng)

    all_models = {**dl_models, **classical_models}

    for model_name, model in all_models.items():
        is_dl = model_name in DL_REGISTRY
        model_type = 'deep_learning' if is_dl else 'classical'
        X_eval = X_test if is_dl else X_test_eng
        X_tr = X_train if is_dl else X_train_eng
        enf = enforcer_dl if is_dl else enforcer_clf
        sna_base = sna_base_dl if is_dl else sna_base_clf
        thresh = 0.5 if is_dl else thresholds.get(
            SELECTED_MODELS.get(model_name, model_name), 0.5)

        print(f"\n  {model_name}  "
              f"({'scaled' if is_dl else 'unscaled'} input, "
              f"threshold={thresh:.2f})")
        baseline = evaluate(model, X_eval, y_test, model_name,
                            threshold=thresh)
        infer_us = measure_inference_latency(model, X_eval)

        # ── DLSA ──────────────────────────────────────────────────────
        for scale in DLSA_SCALES:
            try:
                dlsa = DataLocationShiftAttack(
                    shift_scale=scale, feature_names=feature_names,
                    gnss_enforcer=enf, seed=42)

                t0 = time.perf_counter()
                X_adv = dlsa.generate(X_eval, y_test)
                atk_gen_us = (time.perf_counter() - t0) / len(X_eval) * 1e6

                adv = evaluate(model, X_adv, y_test, model_name,
                               threshold=thresh)
                d = delta_metrics(baseline, adv)
                asr = attack_success_rate(model, X_eval, X_adv, y_test,
                                          threshold=thresh)
                print(f"    DLSA scale={scale:.2f}  "
                      f"ΔF1={d['delta_f1']:+.4f}  "
                      f"ΔRecall={d['delta_recall']:+.4f}  "
                      f"ASR={asr:.3f}")
                results.append(_row(model_name, model_type, 'DLSA', scale,
                                    thresh, baseline, adv, d, asr,
                                    infer_us, atk_gen_us))
            except Exception as e:
                print(f"    DLSA scale={scale:.2f}: FAILED — {e}")

        # ── SNA ───────────────────────────────────────────────────────
        for eps in SNA_EPSILONS:
            try:
                sna = SimilarityNoiseAttack(epsilon=eps,
                                            gnss_enforcer=enf, seed=42)
                sna._std = sna_base._std

                t0 = time.perf_counter()
                X_adv = sna.generate(X_eval, y_test)
                atk_gen_us = (time.perf_counter() - t0) / len(X_eval) * 1e6

                adv = evaluate(model, X_adv, y_test, model_name,
                               threshold=thresh)
                d = delta_metrics(baseline, adv)
                asr = attack_success_rate(model, X_eval, X_adv, y_test,
                                          threshold=thresh)
                print(f"    SNA  ε={eps:.2f}       "
                      f"ΔF1={d['delta_f1']:+.4f}  "
                      f"ΔRecall={d['delta_recall']:+.4f}  "
                      f"ASR={asr:.3f}")
                results.append(_row(model_name, model_type, 'SNA', eps,
                                    thresh, baseline, adv, d, asr,
                                    infer_us, atk_gen_us))
            except Exception as e:
                print(f"    SNA  ε={eps:.2f}: FAILED — {e}")

        # ── TPA ───────────────────────────────────────────────────────
        for (d_amp, c_amp) in TPA_CONFIGS:
            try:
                tpa = TemporalPatternAttack(
                    doppler_amp=d_amp, cn0_amp=c_amp,
                    feature_names=feature_names,
                    gnss_enforcer=enf, seed=42)

                t0 = time.perf_counter()
                X_adv = tpa.generate(X_eval, y_test)
                atk_gen_us = (time.perf_counter() - t0) / len(X_eval) * 1e6

                adv = evaluate(model, X_adv, y_test, model_name,
                               threshold=thresh)
                d = delta_metrics(baseline, adv)
                asr = attack_success_rate(model, X_eval, X_adv, y_test,
                                          threshold=thresh)
                print(f"    TPA  dop={d_amp:.2f}/cn0={c_amp:.2f}  "
                      f"ΔF1={d['delta_f1']:+.4f}  "
                      f"ΔRecall={d['delta_recall']:+.4f}  "
                      f"ASR={asr:.3f}")
                row = _row(model_name, model_type, 'TPA', d_amp,
                           thresh, baseline, adv, d, asr, infer_us, atk_gen_us)
                row['tpa_cn0_amp'] = c_amp
                results.append(row)
            except Exception as e:
                print(f"    TPA  dop={d_amp:.2f}: FAILED — {e}")

    return results


def _row(model_name, model_type, attack, epsilon, threshold,
         baseline, adv, d, asr, infer_us, atk_gen_us) -> dict:
    """Build a result dict with consistent column names."""
    return {
        'model':      model_name,
        'model_type': model_type,
        'attack':     attack,
        'epsilon':    epsilon,
        'threshold':  threshold,
        'baseline_accuracy':  baseline['accuracy'],
        'baseline_f1':        baseline['f1'],
        'baseline_recall':    baseline['recall'],
        'baseline_precision': baseline['precision'],
        'baseline_auc_roc':   baseline['auc_roc'],
        'adv_accuracy':  adv['accuracy'],
        'adv_f1':        adv['f1'],
        'adv_recall':    adv['recall'],
        'adv_precision': adv['precision'],
        'adv_auc_roc':   adv['auc_roc'],
        **d,
        'attack_success_rate': asr,
        'detector_infer_us':   infer_us,
        'attack_gen_us':        atk_gen_us,
        'latency_ratio':        atk_gen_us / max(infer_us, 1e-9),
    }


# --- Visualisation ---

def _fig_dir(base: Path) -> Path:
    d = base / 'figures' / 'adversarial'
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_robust_accuracy_curves(results_df: pd.DataFrame, save_dir: Path):
    """A. Robust accuracy curves — DL models under gradient attacks."""
    grad = results_df[results_df['attack'].isin(['FGSM', 'PGD'])]
    if grad.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, ylabel in [
        (axes[0], 'adv_accuracy', 'Robust Accuracy'),
        (axes[1], 'adv_recall',   'Robust Recall'),
    ]:
        for atk, grp in grad.groupby('attack'):
            pivot = (grp.groupby(['model', 'epsilon'])[metric]
                     .mean().unstack('epsilon'))
            means = pivot.mean(axis=0)
            stds = pivot.std(axis=0)
            ax.plot(means.index, means.values, marker='o', lw=2, label=atk)
            ax.fill_between(means.index,
                            means.values - stds.values,
                            means.values + stds.values, alpha=0.15)
        if metric == 'adv_accuracy':
            bl = grad.groupby('model')['baseline_accuracy'].first().mean()
            ax.axhline(bl, ls='--', color='grey', lw=1.5,
                       label=f'Clean baseline ({bl:.3f})')
        ax.set_xlabel('ε (normalised, L-∞)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{ylabel} vs Perturbation Budget', fontsize=13)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Robust Accuracy Curves — DL Models (White-Box)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    out = save_dir / 'robust_accuracy_curves.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {out.name}")


def plot_asr_comparison(results_df: pd.DataFrame, save_dir: Path):
    """B. Attack success rate at ε=0.10, all attacks and models."""
    mid = results_df[
        results_df['epsilon'].apply(lambda x: abs(x - 0.10) < 1e-6) &
        results_df['attack_success_rate'].notna()
    ]
    if mid.empty:
        return
    pivot = mid.pivot_table(values='attack_success_rate',
                            index='model', columns='attack', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(14, 6))
    pivot.plot(kind='bar', ax=ax, edgecolor='black', width=0.75)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Attack Success Rate', fontsize=12)
    ax.set_title('Attack Success Rate at ε = 0.10', fontsize=13)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend(title='Attack', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out = save_dir / 'attack_success_rate.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {out.name}")


def build_worst_case_table(results_df: pd.DataFrame,
                           save_dir: Path) -> pd.DataFrame:
    """C. Worst-case robustness per model — minimum adversarial recall."""
    if results_df.empty:
        return pd.DataFrame()
    rows = []
    for model_name, grp in results_df.groupby('model'):
        worst_row = grp.loc[grp['adv_recall'].idxmin()]
        rows.append({
            'model':              model_name,
            'model_type':         grp['model_type'].iloc[0],
            'clean_f1':           grp['baseline_f1'].iloc[0],
            'clean_recall':       grp['baseline_recall'].iloc[0],
            'worst_adv_recall':   float(grp['adv_recall'].min()),
            'worst_adv_f1':       float(grp['adv_f1'].min()),
            'worst_adv_accuracy': float(grp['adv_accuracy'].min()),
            'worst_attack':       (f"{worst_row['attack']}  "
                                   f"ε={worst_row['epsilon']:.2f}"),
            'mean_delta_f1':      float(grp['delta_f1'].mean()),
            'mean_delta_recall':  float(grp['delta_recall'].mean()),
            'mean_asr':           float(grp['attack_success_rate'].mean()),
        })
    wc_df = (pd.DataFrame(rows)
             .sort_values('worst_adv_recall', ascending=False)
             .reset_index(drop=True))

    out_csv = TABLES_DIR / 'worst_case_robustness.csv'
    wc_df.to_csv(out_csv, index=False)
    print(f"  ✓ Worst-case table: {out_csv.name}")

    metric_cols = ['clean_recall', 'worst_adv_recall',
                   'worst_adv_f1', 'mean_asr']
    heat_data = wc_df.set_index('model')[metric_cols]
    fig, ax = plt.subplots(figsize=(10, max(6, len(wc_df) * 0.5)))
    sns.heatmap(heat_data, annot=True, fmt='.3f', cmap='RdYlGn',
                ax=ax, vmin=0, vmax=1, linewidths=0.5,
                cbar_kws={'label': 'Score'})
    ax.set_title('Worst-Case Robustness Across All Attacks',
                 fontsize=13, fontweight='bold')
    ax.set_xticklabels(
        ['Clean Recall', 'Worst Adv. Recall',
         'Worst Adv. F1', 'Mean ASR'],
        rotation=20, ha='right')
    fig.tight_layout()
    out = save_dir / 'worst_case_heatmap.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {out.name}")
    return wc_df


def plot_latency_analysis(results_df: pd.DataFrame, save_dir: Path):
    """D. Deployment feasibility — detector inference vs attack generation."""
    lat_df = results_df[
        ['model', 'attack', 'model_type',
         'detector_infer_us', 'attack_gen_us', 'latency_ratio']
    ].dropna(subset=['attack_gen_us'])
    if lat_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    det = results_df.groupby(
        'model')['detector_infer_us'].first().sort_values()
    colors = ['steelblue' if m in DL_REGISTRY else 'darkorange'
              for m in det.index]
    ax.barh(range(len(det)), det.values, color=colors, edgecolor='black')
    ax.set_yticks(range(len(det)))
    ax.set_yticklabels(det.index, fontsize=9)
    ax.set_xlabel('Inference latency (µs/sample)', fontsize=11)
    ax.set_title('Detector Inference Latency', fontsize=12)
    for i, v in enumerate(det.values):
        ax.text(v + max(det.values) * 0.01, i,
                f'{v:.1f}', va='center', fontsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='steelblue', label='DL'),
                       Patch(facecolor='darkorange', label='Classical')],
              fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    ax = axes[1]
    atk_colors = {
        'FGSM': 'steelblue', 'PGD': 'navy',
        'DLSA': 'forestgreen', 'SNA': 'seagreen', 'TPA': 'mediumseagreen',
    }
    for atk, grp in lat_df.groupby('attack'):
        ax.scatter(grp['detector_infer_us'], grp['attack_gen_us'],
                   label=atk, alpha=0.7, s=60,
                   color=atk_colors.get(atk, 'grey'), edgecolors='black')
    lim = max(lat_df[['detector_infer_us', 'attack_gen_us']].max()) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5,
            label='Attack speed = Detection speed')
    ax.set_xlabel('Detector inference (µs/sample)', fontsize=11)
    ax.set_ylabel('Attack generation (µs/sample)', fontsize=11)
    ax.set_title('Latency Trade-off', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Deployment Feasibility: Latency Analysis',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    out = save_dir / 'latency_analysis.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {out.name}")


def plot_robustness_summary(results_df: pd.DataFrame, save_dir: Path):
    """Three-panel summary: gradient / transfer / feature-space attacks."""
    grad_df = results_df[results_df['attack'].isin(['FGSM', 'PGD'])]
    tran_df = results_df[results_df['attack'].isin(['FGSM-Transfer',
                                                    'PGD-Transfer'])]
    feat_df = results_df[results_df['attack'].isin(['DLSA', 'SNA', 'TPA'])]
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    ax = axes[0]
    for atk, grp in grad_df.groupby('attack'):
        means = grp.groupby('epsilon')['delta_f1'].mean()
        stds = grp.groupby('epsilon')['delta_f1'].std()
        ax.plot(means.index, means.values, marker='o', lw=2, label=atk)
        ax.fill_between(means.index, means - stds, means + stds, alpha=0.15)
    ax.set_xlabel('ε', fontsize=12)
    ax.set_ylabel('Mean ΔF1', fontsize=12)
    ax.set_title('Gradient Attacks — DL Models', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    sub = tran_df[tran_df['epsilon'].apply(lambda x: abs(x - 0.10) < 1e-6)]
    if not sub.empty:
        pivot = sub.pivot_table(values='delta_recall', index='model',
                                columns='attack', aggfunc='mean')
        pivot.plot(kind='barh', ax=ax, edgecolor='black')
        ax.set_xlabel('ΔRecall', fontsize=12)
        ax.set_title('Transfer Attacks — Classical (ε=0.10)', fontsize=12)
        ax.axvline(0, color='black', lw=0.8, ls='--')
        ax.grid(axis='x', alpha=0.3)

    ax = axes[2]
    rows = []
    for atk, eps in [('DLSA', 0.10), ('SNA', 0.10), ('TPA', 0.10)]:
        sub = feat_df[
            (feat_df['attack'] == atk) &
            feat_df['epsilon'].apply(lambda x: abs(x - eps) < 1e-6)
        ]
        if not sub.empty:
            rows.append(sub[['model', 'attack', 'delta_f1']])
    if rows:
        combined = pd.concat(rows)
        pivot = combined.pivot_table(values='delta_f1', index='model',
                                     columns='attack', aggfunc='mean')
        pivot.plot(kind='barh', ax=ax, edgecolor='black')
        ax.set_xlabel('ΔF1', fontsize=12)
        ax.set_title('Feature-Space Attacks — All Models', fontsize=12)
        ax.axvline(0, color='black', lw=0.8, ls='--')
        ax.grid(axis='x', alpha=0.3)

    fig.suptitle('Adversarial Robustness — TEXBAT',
                 fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    out = save_dir / 'robustness_summary.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {out.name}")


# --- Console summaries ---

def print_summaries(results_df: pd.DataFrame, wc_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("A. ROBUST ACCURACY — GRADIENT ATTACKS (mean over DL models)")
    print("=" * 70)
    grad = results_df[results_df['attack'].isin(['FGSM', 'PGD'])]
    if not grad.empty:
        tbl = (grad.groupby(['attack', 'epsilon'])
               [['adv_accuracy', 'adv_recall', 'adv_f1']]
               .mean().round(4))
        print(tbl.to_string())

    print("\n" + "=" * 70)
    print("B. ATTACK SUCCESS RATE — all attacks at ε=0.10")
    print("=" * 70)
    mid = results_df[
        results_df['epsilon'].apply(lambda x: abs(x - 0.10) < 1e-6)]
    if not mid.empty:
        tbl = (mid.groupby(['model', 'attack'])['attack_success_rate']
               .mean().unstack('attack').round(3))
        print(tbl.to_string())

    print("\n" + "=" * 70)
    print("C. WORST-CASE ROBUSTNESS (sorted by worst adversarial recall)")
    print("=" * 70)
    if not wc_df.empty:
        cols = ['model', 'model_type', 'clean_recall', 'worst_adv_recall',
                'worst_adv_f1', 'worst_attack', 'mean_asr']
        print(wc_df[cols].to_string(index=False))

    print("\n" + "=" * 70)
    print("D. LATENCY SUMMARY — detector inference vs attack generation (µs/sample)")
    print("=" * 70)
    lat = (results_df.dropna(subset=['attack_gen_us'])
           .groupby(['model', 'attack'])
           [['detector_infer_us', 'attack_gen_us', 'latency_ratio']]
           .mean().round(2))
    if not lat.empty:
        print(lat.to_string())
        print("\nLatency ratio > 1 means attack generation is SLOWER than detection")
        print("(real-time adaptive attacks computationally infeasible for that pair)")


def main():
    parser = argparse.ArgumentParser(
        description='Adversarial robustness evaluation for GNSS spoofing detectors')
    parser.add_argument('--epsilons', type=float, nargs='+', default=EPSILONS,
                        help='L-inf epsilon values for gradient attacks')
    parser.add_argument('--quick', action='store_true',
                        help='Use 2000 test samples for fast sanity check (~10 min)')
    args = parser.parse_args()

    create_directories()
    print("=" * 70)
    print("ADVERSARIAL ATTACK EVALUATION")
    print("=" * 70)

    # Data loading
    (X_train,     X_val,     X_test,
     X_train_eng, X_val_eng, X_test_eng,
     y_train, y_val, y_test,
     feat_names, scaler) = load_and_preprocess()

    if args.quick:
        X_test,     y_test = X_test[:2000],     y_test[:2000]
        X_test_eng = X_test_eng[:2000]
        X_train_sub = X_train[:5000]
        X_train_eng_sub = X_train_eng[:5000]
        print(f"  [--quick] {len(X_test)} test samples | "
              f"{len(X_train_sub)} train samples for SNA fit")
    else:
        X_train_sub = X_train
        X_train_eng_sub = X_train_eng

    # Fit constraint enforcers (one per data space)
    enforcer_dl = GNSSConstraintEnforcer(feat_names)
    enforcer_dl.fit(X_train_sub, feat_names)          # fitted on scaled data

    enforcer_clf = GNSSConstraintEnforcer(feat_names)
    enforcer_clf.fit(X_train_eng_sub, feat_names)     # fitted on unscaled data

    # Load optimal thresholds
    print("\n--- Loading optimal thresholds ---")
    thresholds = load_optimal_thresholds()

    # Load models
    print("\n" + "=" * 70)
    print("LOADING MODELS")
    print("=" * 70)
    print("\nClassical models (expect UNSCALED input):")
    classical_models = load_classical_models()
    print("\nDeep learning models (expect SCALED input):")
    dl_models = load_dl_models(input_dim=X_test.shape[1])

    if not classical_models and not dl_models:
        print("\n❌ No models found — run baseline experiments first.")
        return

    print(
        f"\n✓ {len(classical_models)} classical | {len(dl_models)} DL models loaded")

    # Preprocessing diagnostic
    if classical_models:
        ok = run_preprocessing_diagnostic(
            classical_models, X_test_eng, y_test, thresholds)
        if not ok:
            print("❌ Halting — fix model loading before running attacks.")
            return

    all_results = []

    # Gradient-based attacks on DL models
    if dl_models:
        print("\n" + "=" * 70)
        print("GRADIENT-BASED ATTACKS  (DL models, white-box, scaled space)")
        print("=" * 70)
        all_results.extend(
            run_gradient_attacks(
                dl_models, X_test, y_test,
                args.epsilons, feat_names, enforcer_dl))

    # Transfer attacks: DL surrogate → classical models
    if dl_models and classical_models:
        print("\n" + "=" * 70)
        print("TRANSFER ATTACKS  (surrogate=CNN-1D → classical, unscaled)")
        print("=" * 70)
        all_results.extend(
            run_transfer_attacks(
                dl_models, classical_models,
                X_test, X_test_eng, y_test,
                args.epsilons, feat_names,
                enforcer_dl, scaler, thresholds))

    # Feature-space attacks: all models
    print("\n" + "=" * 70)
    print("FEATURE-SPACE ATTACKS  (all models, routed by type)")
    print("  DL models  → scaled input + enforcer_dl")
    print("  Classical  → unscaled input + enforcer_clf")
    print("=" * 70)
    all_results.extend(
        run_feature_space_attacks(
            dl_models, classical_models,
            X_train_sub,     X_test,
            X_train_eng_sub, X_test_eng,
            y_test, feat_names,
            enforcer_dl, enforcer_clf, thresholds))

    if not all_results:
        print("\n❌ No results generated.")
        return

    # Save results
    results_df = pd.DataFrame(all_results)
    out_path = TABLES_DIR / 'adversarial_attack_results.csv'
    results_df.to_csv(out_path, index=False)
    print(f"\n✓ Full results: {out_path}")

    # Tables and figures
    fig_dir = _fig_dir(TABLES_DIR.parent)
    wc_df = build_worst_case_table(results_df, fig_dir)

    print_summaries(results_df, wc_df)

    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    plot_robust_accuracy_curves(results_df, fig_dir)
    plot_asr_comparison(results_df, fig_dir)
    plot_latency_analysis(results_df, fig_dir)
    plot_robustness_summary(results_df, fig_dir)

    print("\n" + "=" * 70)
    print("ADVERSARIAL EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Results  → {out_path}")
    print(f"Tables   → {TABLES_DIR / 'worst_case_robustness.csv'}")
    print(f"Figures  → {fig_dir}")
    print("\nRun with --quick for fast sanity check (~10 min, 2000 samples).")


if __name__ == "__main__":
    main()
