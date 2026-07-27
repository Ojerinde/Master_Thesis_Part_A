"""
Statistical Analysis — bootstrap CIs, McNemar's test, Wilcoxon signed-rank,
Friedman test, Cohen's d for adversarial robustness evaluation.
All p-values reported to 4 d.p., Bonferroni-Holm corrected where applicable.

Usage:
    python -m experiments.04_statistical_analysis
"""

from config.paths import CLASSICAL_MODELS, DL_MODELS, TABLES_DIR, create_directories
from config.model_configs import get_config
from models.deep_learning import (
    CNN1DModel, LSTMModel, BiLSTMModel,
    CNNLSTMModel, TransformerModel, TCNModel,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from data.feature_engineering import SafeFeatureEngineer
from data.loader import load_texbat
import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
import scipy.stats as stats
from itertools import combinations
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
warnings.filterwarnings('ignore')


create_directories()

# ---------------------------------------------------------------------------
# Model registry — must match 03_adversarial_evaluation.py
# ---------------------------------------------------------------------------
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

SIGNIFICANCE_LEVELS = {0.001: '***', 0.01: '**', 0.05: '*', 1.0: 'ns'}


def sig_stars(p):
    for threshold, stars in SIGNIFICANCE_LEVELS.items():
        if p < threshold:
            return stars
    return 'ns'


# --- Section 1: Data loading ---

def load_data():
    print("\n" + "=" * 70)
    print("DATA LOADING (matching 03_adversarial_evaluation.py split)")
    print("=" * 70)

    df, loader = load_texbat(verbose=False, validate=False)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in ['label', 'Label', 'channel', 'scenario']:
        if col in numeric_cols:
            numeric_cols.remove(col)
    if 'Label' in df.columns:
        df = df.rename(columns={'Label': 'label'})

    y = df['label'].values.astype(int)
    X_raw = df[numeric_cols].values

    X_temp, X_test_raw, y_temp, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp)

    engineer = SafeFeatureEngineer(verbose=False)
    X_train_eng, X_val_eng, feat_names = engineer.fit_transform(
        X_train_raw, X_val_raw, feature_names=numeric_cols)
    X_test_eng = engineer.transform(X_test_raw)

    X_train_eng = X_train_eng.astype(np.float32)
    X_test_eng = X_test_eng.astype(np.float32)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_eng).astype(np.float32)
    X_test = scaler.transform(X_test_eng).astype(np.float32)

    print(f"✓ Test set: {X_test.shape[0]:,} samples  |  "
          f"Spoofing prevalence: {y_test.mean():.3f}")
    return X_test, X_test_eng, y_test


# --- Section 2: Model loading & prediction ---

def load_optimal_thresholds():
    path = TABLES_DIR / 'optimal_thresholds.csv'
    if not path.exists():
        print("  ⚠ optimal_thresholds.csv not found — using 0.50 for all models")
        return {}
    df = pd.read_csv(path)
    return dict(zip(df['model'], df['optimal_threshold']))


def _proba_classical(model, X):
    X = X.astype(np.float64)
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def _proba_dl(model_wrapper, X):
    device = next(model_wrapper.model.parameters()).device
    model_wrapper.model.eval()
    with torch.no_grad():
        t = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model_wrapper.model(t)
        return torch.sigmoid(logits).cpu().numpy().flatten()


def load_models_and_predict(X_test_scaled, X_test_unscaled, y_test, thresholds):
    """
    Load all models, generate clean predictions, return dict of binary arrays.
    """
    print("\n" + "=" * 70)
    print("LOADING MODELS AND GENERATING CLEAN PREDICTIONS")
    print("=" * 70)

    preds = {}   # model_name → binary predictions (0/1) on test set
    probas = {}  # model_name → probability scores

    # ---- Classical models ----
    for display_name, file_name in SELECTED_CLASSICAL.items():
        path = CLASSICAL_MODELS / f"{file_name}.joblib"
        if not path.exists():
            print(f"  ⚠ Not found: {path}")
            continue
        model = joblib.load(path)
        thresh = thresholds.get(file_name, 0.5)
        try:
            proba = _proba_classical(model, X_test_unscaled)
            pred = (proba >= thresh).astype(int)
            preds[display_name] = pred
            probas[display_name] = proba
            acc = (pred == y_test).mean()
            f1 = _f1(pred, y_test)
            print(f"  ✓ {display_name:20s}  threshold={thresh:.2f}  "
                  f"Acc={acc:.4f}  F1={f1:.4f}")
        except Exception as exc:
            print(f"  ⚠ {display_name}: {exc}")

    # ---- DL models ----
    input_dim = X_test_scaled.shape[1]
    for name, (ModelClass, config_name) in DL_REGISTRY.items():
        path = DL_MODELS / f"{config_name}.pt"
        if not path.exists():
            print(f"  ⚠ DL model not found: {path}")
            continue
        try:
            config = get_config(config_name)
            config['input_dim'] = input_dim
            wrapper = ModelClass(input_dim=input_dim, config=config)
            wrapper.build_model()
            wrapper.model.load_state_dict(
                torch.load(str(path), map_location='cpu'))
            wrapper.model.eval()
            wrapper.is_trained = True
            proba = _proba_dl(wrapper, X_test_scaled)
            pred = (proba >= 0.5).astype(int)
            preds[name] = pred
            probas[name] = proba
            acc = (pred == y_test).mean()
            f1 = _f1(pred, y_test)
            print(f"  ✓ {name:20s}  threshold=0.50  "
                  f"Acc={acc:.4f}  F1={f1:.4f}")
        except Exception as exc:
            print(f"  ⚠ {name}: {exc}")

    print(f"\n✓ {len(preds)} models loaded successfully")
    return preds, probas


# --- Section 3: Metric helpers ---

def _f1(pred, y):
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    p = tp / (tp + fp + 1e-12)
    r = tp / (tp + fn + 1e-12)
    return 2 * p * r / (p + r + 1e-12)


def _recall(pred, y):
    tp = ((pred == 1) & (y == 1)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    return tp / (tp + fn + 1e-12)


def _precision(pred, y):
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    return tp / (tp + fp + 1e-12)


# --- Section 4: Bootstrap CIs (BCa) ---

def bca_bootstrap_ci(scores, stat_fn, n_bootstrap=10_000, confidence=0.95,
                     seed=42):
    """
    Bias-corrected and accelerated (BCa) bootstrap CI.

    Parameters
    ----------
    scores     : (n,) array of per-sample binary correctness or metric
    stat_fn    : function(scores_subset) → scalar metric
    n_bootstrap: number of bootstrap resamples
    confidence : nominal coverage
    seed       : for reproducibility

    Returns
    -------
    (point_estimate, lower, upper)

    Reference: Efron & Hastie (2016), Ch. 11.
    """
    rng = np.random.default_rng(seed)
    n = len(scores)
    theta = stat_fn(scores)

    boot_stats = np.array([
        stat_fn(scores[rng.integers(0, n, n)])
        for _ in range(n_bootstrap)
    ])

    # Bias-correction factor z0
    z0 = stats.norm.ppf(np.mean(boot_stats < theta))

    # Acceleration factor a — jackknife
    jack_stats = np.array([
        stat_fn(np.delete(scores, i)) for i in range(n)
    ])
    jack_mean = jack_stats.mean()
    num = np.sum((jack_mean - jack_stats) ** 3)
    denom = 6.0 * (np.sum((jack_mean - jack_stats) ** 2) ** 1.5)
    a = num / (denom + 1e-12)

    alpha = 1 - confidence
    z_lo = stats.norm.ppf(alpha / 2)
    z_hi = stats.norm.ppf(1 - alpha / 2)

    alpha_lo = stats.norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    alpha_hi = stats.norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))

    lower = float(np.percentile(boot_stats, 100 * alpha_lo))
    upper = float(np.percentile(boot_stats, 100 * alpha_hi))
    return theta, lower, upper


def run_bootstrap_cis(preds, y_test):
    """
    Table 1: BCa bootstrap CIs for F1, Recall, Precision per model.
    Required by IEEE for any performance claim.
    """
    print("\n" + "=" * 70)
    print("TABLE 1 — BOOTSTRAP CONFIDENCE INTERVALS  (BCa, 95%, n=10 000)")
    print("  H0: metric = chance baseline (F1=0.857 for all-spoofing predictor)")
    print("=" * 70)

    rows = []
    for name, pred in preds.items():
        correct = (pred == y_test).astype(float)
        f1_val = _f1(pred, y_test)
        rec_val = _recall(pred, y_test)
        pre_val = _precision(pred, y_test)

        # Bootstrap over test samples
        # For F1/Recall/Precision we bootstrap the sample indices
        y_arr = y_test.astype(float)
        indices = np.arange(len(y_test))

        def f1_from_idx(idx):
            return _f1(pred[idx.astype(int)], y_arr[idx.astype(int)])

        def rec_from_idx(idx):
            return _recall(pred[idx.astype(int)], y_arr[idx.astype(int)])

        def pre_from_idx(idx):
            return _precision(pred[idx.astype(int)], y_arr[idx.astype(int)])

        _, f1_lo,  f1_hi = bca_bootstrap_ci(indices, f1_from_idx)
        _, rec_lo, rec_hi = bca_bootstrap_ci(indices, rec_from_idx)
        _, pre_lo, pre_hi = bca_bootstrap_ci(indices, pre_from_idx)

        rows.append({
            'Model':          name,
            'F1':             round(f1_val, 4),
            'F1_CI_lower':    round(f1_lo,  4),
            'F1_CI_upper':    round(f1_hi,  4),
            'Recall':         round(rec_val, 4),
            'Recall_CI_lower': round(rec_lo,  4),
            'Recall_CI_upper': round(rec_hi,  4),
            'Precision':      round(pre_val, 4),
            'Precision_CI_lower': round(pre_lo, 4),
            'Precision_CI_upper': round(pre_hi, 4),
        })

        print(f"  {name:20s}  "
              f"F1={f1_val:.4f} [{f1_lo:.4f}, {f1_hi:.4f}]  "
              f"Rec={rec_val:.4f} [{rec_lo:.4f}, {rec_hi:.4f}]  "
              f"Pre={pre_val:.4f} [{pre_lo:.4f}, {pre_hi:.4f}]")

    df = pd.DataFrame(rows).sort_values(
        'F1', ascending=False).reset_index(drop=True)
    path = TABLES_DIR / 'table1_bootstrap_cis.csv'
    df.to_csv(path, index=False)
    print(f"\n✓ Saved: {path}")
    return df


# --- Section 5: McNemar's test ---

def mcnemar_exact(pred_a, pred_b, y):
    """
    McNemar's exact test (mid-p corrected for continuity).
    Tests H0: both models have equal error rates.

    b = A correct,  B wrong
    c = A wrong,    B correct

    Under H0, b ~ Binomial(b+c, 0.5).
    p-value = 2 * P(X >= max(b,c)) using exact binomial.

    Returns: (p_value, odds_ratio)
    Odds ratio = b/c  (>1 means model A is better, <1 means B is better)
    """
    correct_a = (pred_a == y)
    correct_b = (pred_b == y)
    b = int((correct_a & ~correct_b).sum())  # A right, B wrong
    c = int((~correct_a & correct_b).sum())  # A wrong, B right

    if b + c == 0:
        return 1.0, 1.0   # identical predictions

    n = b + c
    lo = min(b, c)
    # Mid-p exact McNemar (Fagerland et al., 2013)
    p_val = 2 * (stats.binom.sf(lo, n, 0.5) +
                 0.5 * stats.binom.pmf(lo, n, 0.5))
    p_val = min(p_val, 1.0)
    odds = (b / c) if c > 0 else float('inf')
    return p_val, odds


def run_mcnemar(preds, y_test):
    """
    Table 2: Pairwise McNemar's test with Bonferroni-Holm correction.
    """
    print("\n" + "=" * 70)
    print("TABLE 2 — MCNEMAR'S TEST  (pairwise, mid-p exact, Bonferroni-Holm)")
    print("  H0: model A and model B have equal error rates on test set")
    print("=" * 70)

    model_names = list(preds.keys())
    pairs = list(combinations(model_names, 2))
    rows = []

    for a, b in pairs:
        p, odds = mcnemar_exact(preds[a], preds[b], y_test)
        rows.append({
            'Model_A':   a,
            'Model_B':   b,
            'p_raw':     round(p, 6),
            'odds_ratio': round(odds, 4),
        })

    # Bonferroni-Holm correction
    raw_ps = np.array([r['p_raw'] for r in rows])
    order = np.argsort(raw_ps)
    m = len(raw_ps)
    p_adj = np.empty(m)
    for rank, idx in enumerate(order):
        p_adj[idx] = min(1.0, raw_ps[idx] * (m - rank))
    # Monotone enforcement
    for i in range(len(order) - 2, -1, -1):
        p_adj[order[i]] = min(p_adj[order[i]], p_adj[order[i + 1]])

    for i, r in enumerate(rows):
        r['p_adjusted'] = round(p_adj[i], 6)
        r['significant'] = p_adj[i] < 0.05
        r['stars'] = sig_stars(p_adj[i])

    df = pd.DataFrame(rows).sort_values('p_adjusted')
    path = TABLES_DIR / 'table2_mcnemar_pairwise.csv'
    df.to_csv(path, index=False)

    sig_pairs = df[df['significant']]
    print(f"  {len(sig_pairs)}/{len(df)} pairs significantly different "
          f"(Bonferroni-Holm α=0.05):\n")
    print(f"  {'Model A':22s}  {'Model B':22s}  "
          f"{'p_adj':>10s}  {'OR':>8s}  {'Sig':>4s}")
    print(f"  {'-'*22}  {'-'*22}  {'-'*10}  {'-'*8}  {'-'*4}")
    for _, row in df.iterrows():
        print(f"  {row['Model_A']:22s}  {row['Model_B']:22s}  "
              f"{row['p_adjusted']:10.4f}  {row['odds_ratio']:8.3f}  "
              f"{row['stars']:>4s}")

    print(f"\n✓ Saved: {path}")
    return df


# --- Section 6: Wilcoxon signed-rank ---

def _rank_biserial(w_stat, n):
    """
    Rank-biserial correlation r from Wilcoxon W statistic.
    r = 1 - 4W / (n*(n+1))   [matched pairs version]
    Interpretation: |r| ≥ 0.1 small, ≥ 0.3 medium, ≥ 0.5 large.
    """
    return 1 - (4 * w_stat) / (n * (n + 1))


def run_wilcoxon_adversarial(adv_csv_path):
    """
    Table 3: Wilcoxon signed-rank — does each attack type significantly
    degrade F1?  Each model is one paired observation (clean F1, adv F1).

    H0: median ΔF1 = 0  (attack has no effect)
    H1: median ΔF1 > 0  (attack degrades F1)

    Performed separately for DL models and classical models, and combined.
    """
    print("\n" + "=" * 70)
    print("TABLE 3 — WILCOXON SIGNED-RANK  (clean vs adversarial F1)")
    print("  H0: attack causes no F1 degradation  (one-sided, α=0.05)")
    print("=" * 70)

    if not adv_csv_path.exists():
        print("  ⚠ adversarial_attack_results.csv not found — skipping")
        return None

    adv = pd.read_csv(adv_csv_path)

    # Expected columns: model, model_type, attack, epsilon,
    #                   baseline_f1, adv_f1
    required = {'model', 'model_type', 'attack',
                'epsilon', 'baseline_f1', 'adv_f1'}
    if not required.issubset(adv.columns):
        missing = required - set(adv.columns)
        print(f"  ⚠ Missing columns: {missing}")
        print(f"    Available: {list(adv.columns)}")
        print("  ⚠ Skipping Wilcoxon — check adversarial_attack_results.csv schema")
        return None

    adv['delta_f1'] = adv['baseline_f1'] - adv['adv_f1']
    rows = []

    attack_groups = adv.groupby(['attack', 'epsilon'])
    for (attack, eps), grp in attack_groups:
        for model_type in ['deep_learning', 'classical', 'all']:
            if model_type == 'all':
                sub = grp
            else:
                sub = grp[grp['model_type'] == model_type]

            deltas = sub['delta_f1'].dropna().values
            n = len(deltas)
            if n < 4:
                continue   # Wilcoxon requires n ≥ 4

            try:
                # one-sided: H1: degradation > 0
                w_stat, p_two = stats.wilcoxon(deltas, alternative='greater',
                                               zero_method='pratt')
                r_eff = _rank_biserial(w_stat, n)
                rows.append({
                    'attack':          f"{attack} ε={eps}",
                    'model_group':     model_type,
                    'n_models':        n,
                    'median_delta_f1': round(float(np.median(deltas)), 4),
                    'mean_delta_f1':   round(float(np.mean(deltas)), 4),
                    'W_statistic':     round(w_stat, 2),
                    'p_value':         round(p_two, 6),
                    'effect_r':        round(r_eff, 3),
                    'significant':     p_two < 0.05,
                    'stars':           sig_stars(p_two),
                })
            except Exception:
                pass

    if not rows:
        print("  ⚠ No valid Wilcoxon tests could be run")
        return None

    df = pd.DataFrame(rows).sort_values(['attack', 'model_group'])
    path = TABLES_DIR / 'table3_wilcoxon_adversarial.csv'
    df.to_csv(path, index=False)

    # Print DL results first, then classical
    for mg in ['deep_learning', 'classical', 'all']:
        sub = df[df['model_group'] == mg]
        if sub.empty:
            continue
        label = {'deep_learning': 'DL models', 'classical': 'Classical',
                 'all': 'All models'}[mg]
        print(f"\n  {label}:")
        print(f"  {'Attack':22s}  {'n':>3}  {'Median ΔF1':>10}  "
              f"{'W':>8}  {'p':>8}  {'r':>6}  {'Sig':>4}")
        print(f"  {'-'*22}  {'-'*3}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*4}")
        for _, row in sub.iterrows():
            print(f"  {row['attack']:22s}  {row['n_models']:3d}  "
                  f"{row['median_delta_f1']:10.4f}  "
                  f"{row['W_statistic']:8.2f}  {row['p_value']:8.4f}  "
                  f"{row['effect_r']:6.3f}  {row['stars']:>4s}")

    print(f"\n✓ Saved: {path}")
    return df


# --- Section 7: Friedman test ---

def run_friedman_attack_comparison(adv_csv_path):
    """
    Table 4: Friedman test — are different attack types significantly
    different in their effectiveness (ΔF1)?

    Null: all attacks produce equal degradation.
    If significant, post-hoc Wilcoxon with Bonferroni-Holm.

    Models = blocks, attack types = treatments.
    Performed on white-box attacks for DL and transfer attacks for classical.
    """
    print("\n" + "=" * 70)
    print("TABLE 4 — FRIEDMAN TEST  (attack type comparison)")
    print("  H0: all attack types cause equal degradation across models")
    print("=" * 70)

    if not adv_csv_path.exists():
        print("  ⚠ adversarial_attack_results.csv not found — skipping")
        return

    adv = pd.read_csv(adv_csv_path)

    required = {'model', 'model_type', 'attack', 'epsilon', 'delta_f1'}
    if 'delta_f1' not in adv.columns:
        if 'clean_f1' in adv.columns and 'adv_f1' in adv.columns:
            adv['delta_f1'] = adv['clean_f1'] - adv['adv_f1']
        else:
            print("  ⚠ Cannot compute delta_f1 — skipping Friedman")
            return

    results_summary = []

    for (model_type, eps), label in [
        (('deep_learning', 0.10), 'DL white-box attacks ε=0.10'),
        (('classical',     0.10), 'Classical transfer attacks ε=0.10'),
    ]:
        sub = adv[(adv['model_type'] == model_type) & (adv['epsilon'] == eps)]
        if sub.empty:
            print(f"  ⚠ No data for {label}")
            continue

        # Pivot: rows=models, cols=attacks
        pivot = sub.pivot_table(
            index='model', columns='attack', values='delta_f1', aggfunc='mean')
        pivot = pivot.dropna(axis=1, how='any').dropna(axis=0, how='any')

        if pivot.shape[1] < 2 or pivot.shape[0] < 3:
            print(f"  ⚠ Insufficient data for Friedman ({label})")
            continue

        attack_cols = list(pivot.columns)
        data_groups = [pivot[c].values for c in attack_cols]

        chi2, p_friedman = stats.friedmanchisquare(*data_groups)
        n_models = pivot.shape[0]
        n_attacks = pivot.shape[1]
        # Kendall's W effect size
        W_kendall = chi2 / (n_models * (n_attacks - 1))

        print(f"\n  {label}")
        print(f"    Models={n_models}  Attacks={n_attacks}  "
              f"χ²={chi2:.3f}  p={p_friedman:.4f}  "
              f"Kendall W={W_kendall:.3f}  {sig_stars(p_friedman)}")
        print(f"    Attack means (ΔF1):")
        for col in attack_cols:
            print(
                f"      {col:25s}  {pivot[col].mean():.4f} ± {pivot[col].std():.4f}")

        results_summary.append({
            'comparison':  label,
            'n_models':    n_models,
            'n_attacks':   n_attacks,
            'chi2':        round(chi2, 4),
            'p_friedman':  round(p_friedman, 6),
            'kendall_W':   round(W_kendall, 4),
            'significant': p_friedman < 0.05,
            'stars':       sig_stars(p_friedman),
        })

        # Post-hoc: pairwise Wilcoxon with Bonferroni-Holm
        if p_friedman < 0.05:
            pairs = list(combinations(attack_cols, 2))
            print(f"\n    Post-hoc pairwise Wilcoxon (Bonferroni-Holm):")
            ph_rows = []
            for a, b in pairs:
                try:
                    _, p_pair = stats.wilcoxon(pivot[a].values, pivot[b].values,
                                               zero_method='pratt')
                    ph_rows.append((a, b, p_pair))
                except Exception:
                    pass
            # Holm correction
            if ph_rows:
                ph_rows.sort(key=lambda x: x[2])
                m = len(ph_rows)
                for rank, (a, b, p_raw) in enumerate(ph_rows):
                    p_adj = min(1.0, p_raw * (m - rank))
                    print(f"      {a:20s} vs {b:20s}  "
                          f"p_adj={p_adj:.4f}  {sig_stars(p_adj)}")

    if results_summary:
        path = TABLES_DIR / 'table4_friedman_attack_comparison.csv'
        pd.DataFrame(results_summary).to_csv(path, index=False)
        print(f"\n✓ Saved: {path}")


# --- Section 8: Cohen's d (DL vs classical) ---

def run_effect_sizes(adv_csv_path):
    """
    Table 5: Cohen's d comparing DL vs classical adversarial degradation.
    Pooled standard deviation version.

    Interpretation: d ≥ 0.2 small, ≥ 0.5 medium, ≥ 0.8 large.
    """
    print("\n" + "=" * 70)
    print("TABLE 5 — EFFECT SIZES  (Cohen's d: DL vs classical degradation)")
    print("=" * 70)

    if not adv_csv_path.exists():
        print("  ⚠ adversarial_attack_results.csv not found — skipping")
        return

    adv = pd.read_csv(adv_csv_path)
    if 'delta_f1' not in adv.columns:
        if 'clean_f1' in adv.columns and 'adv_f1' in adv.columns:
            adv['delta_f1'] = adv['clean_f1'] - adv['adv_f1']
        else:
            print("  ⚠ Cannot compute delta_f1 — skipping")
            return

    rows = []
    for attack_key, grp in adv.groupby(['attack', 'epsilon']):
        attack, eps = attack_key
        dl_deltas = grp[grp['model_type'] ==
                        'deep_learning']['delta_f1'].dropna()
        clf_deltas = grp[grp['model_type'] == 'classical']['delta_f1'].dropna()

        if len(dl_deltas) < 2 or len(clf_deltas) < 2:
            continue

        mu1, mu2 = dl_deltas.mean(), clf_deltas.mean()
        s1, s2 = dl_deltas.std(ddof=1), clf_deltas.std(ddof=1)
        n1, n2 = len(dl_deltas), len(clf_deltas)
        s_pool = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
        d = (mu1 - mu2) / (s_pool + 1e-12)

        # Mann-Whitney U test (non-parametric companion)
        u_stat, p_mw = stats.mannwhitneyu(
            dl_deltas, clf_deltas, alternative='greater')
        # r effect size from U
        r_mw = u_stat / (n1 * n2)

        rows.append({
            'attack':          f"{attack} ε={eps}",
            'DL_mean_deltaF1': round(mu1, 4),
            'DL_std':          round(s1, 4),
            'CLF_mean_deltaF1': round(mu2, 4),
            'CLF_std':         round(s2, 4),
            'Cohen_d':         round(d, 3),
            'MannWhitney_U':   round(u_stat, 1),
            'MannWhitney_p':   round(p_mw, 4),
            'r_effect':        round(r_mw, 3),
            'stars':           sig_stars(p_mw),
        })
        print(f"  {attack} ε={eps:4.2f}:  "
              f"DL ΔF1={mu1:.4f}  CLF ΔF1={mu2:.4f}  "
              f"d={d:.3f}  U_p={p_mw:.4f}  {sig_stars(p_mw)}")

    if rows:
        df = pd.DataFrame(rows)
        path = TABLES_DIR / 'table5_effect_sizes.csv'
        df.to_csv(path, index=False)
        print(f"\n✓ Saved: {path}")
        print("\n  Cohen's d interpretation: ≥0.2 small, ≥0.5 medium, ≥0.8 large")
        print("  DL > Classical ΔF1 means DL models degrade more under that attack")


# --- Section 9: Adversarial degradation summary ---

def run_degradation_summary(adv_csv_path):
    """
    Table 6: Mean and SD of F1 degradation per attack type, across models.
    Direct source for the 'Results' section of the IEEE paper.
    """
    print("\n" + "=" * 70)
    print("TABLE 6 — ADVERSARIAL DEGRADATION SUMMARY")
    print("  Mean ΔF1 ± SD across models per attack type and model group")
    print("=" * 70)

    if not adv_csv_path.exists():
        print("  ⚠ adversarial_attack_results.csv not found — skipping")
        return

    adv = pd.read_csv(adv_csv_path)
    if 'delta_f1' not in adv.columns:
        if 'clean_f1' in adv.columns and 'adv_f1' in adv.columns:
            adv['delta_f1'] = adv['clean_f1'] - adv['adv_f1']
        else:
            print("  ⚠ Cannot compute delta_f1")
            return

    summary = (
        adv.groupby(['attack', 'epsilon', 'model_type'])['delta_f1']
        .agg(['mean', 'std', 'min', 'max', 'count'])
        .round(4)
        .reset_index()
    )
    summary.columns = ['attack', 'epsilon', 'model_type',
                       'mean_delta_f1', 'std_delta_f1',
                       'min_delta_f1', 'max_delta_f1', 'n_models']

    path = TABLES_DIR / 'table6_degradation_summary.csv'
    summary.to_csv(path, index=False)

    for mtype in ['deep_learning', 'classical']:
        sub = summary[summary['model_type'] == mtype]
        label = 'DL models' if mtype == 'deep_learning' else 'Classical models'
        print(f"\n  {label}:")
        print(f"  {'Attack':22s}  {'ε':>5}  {'Mean ΔF1':>10}  "
              f"{'SD':>8}  {'Min':>8}  {'Max':>8}")
        print(f"  {'-'*22}  {'-'*5}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")
        for _, row in sub.iterrows():
            print(f"  {row['attack']:22s}  {row['epsilon']:5.2f}  "
                  f"{row['mean_delta_f1']:10.4f}  {row['std_delta_f1']:8.4f}  "
                  f"{row['min_delta_f1']:8.4f}  {row['max_delta_f1']:8.4f}")

    print(f"\n✓ Saved: {path}")
    return summary


# --- Section 10: Robustness correlation analysis ---

def run_robustness_correlation(adv_csv_path, baseline_csv_path):
    """
    Table 7: Spearman rank correlation between clean F1 and adversarial
    robustness (worst-case adversarial recall).

    Tests whether better clean models are also more adversarially robust.
    This is a common finding in the literature (often they are not correlated
    or even negatively correlated for DL models).
    """
    print("\n" + "=" * 70)
    print("TABLE 7 — ROBUSTNESS CORRELATION")
    print("  Spearman ρ: clean F1 vs worst-case adversarial recall")
    print("=" * 70)

    worst_path = TABLES_DIR / 'worst_case_robustness.csv'
    if not worst_path.exists():
        print("  ⚠ worst_case_robustness.csv not found — skipping")
        return

    worst = pd.read_csv(worst_path)

    # worst_case_robustness.csv already contains clean_f1, model_type,
    # and worst_adv_recall for all 13 models — no merge needed.
    merged = worst.copy()

    for group in ['all', 'deep_learning', 'classical']:
        if group == 'all':
            sub = merged
        else:
            sub = merged[merged['model_type'] == group]

        if len(sub) < 4:
            continue

        rho, p_spear = stats.spearmanr(
            sub['clean_f1'], sub['worst_adv_recall'])
        label = {'all': 'All models', 'deep_learning': 'DL only',
                 'classical': 'Classical only'}[group]
        print(f"  {label:18s}  n={len(sub):2d}  "
              f"Spearman ρ={rho:+.4f}  p={p_spear:.4f}  {sig_stars(p_spear)}")

    corr_rows = []
    for group in ['all', 'deep_learning', 'classical']:
        sub = merged if group == 'all' else merged[merged['model_type'] == group]
        if len(sub) < 4:
            continue
        rho, p = stats.spearmanr(sub['clean_f1'], sub['worst_adv_recall'])
        corr_rows.append({'group': group, 'n': len(sub),
                          'spearman_rho': round(rho, 4),
                          'p_value': round(p, 4),
                          'significant': p < 0.05,
                          'stars': sig_stars(p)})

    path = TABLES_DIR / 'table7_robustness_correlation.csv'
    pd.DataFrame(corr_rows).to_csv(path, index=False)
    print(f"\n✓ Saved: {path}")
    print("  ρ > 0: better clean model → better robustness")
    print("  ρ < 0: better clean model → worse robustness (common in DL)")


def main():
    print("=" * 70)
    print("STATISTICAL ANALYSIS — IEEE ADVERSARIAL ROBUSTNESS EVALUATION")
    print("=" * 70)
    print("\nStatistical tests:")
    print("  1. Bootstrap CIs (BCa, 95%, n=10 000)  — Table 1")
    print("  2. McNemar pairwise (mid-p, Bonferroni-Holm) — Table 2")
    print("  3. Wilcoxon signed-rank clean vs adversarial — Table 3")
    print("  4. Friedman attack-type comparison + post-hoc — Table 4")
    print("  5. Effect sizes (Cohen's d, Mann-Whitney r) — Table 5")
    print("  6. Degradation summary (mean ΔF1 ± SD) — Table 6")
    print("  7. Spearman correlation clean F1 vs robustness — Table 7")

    adv_path = TABLES_DIR / 'adversarial_attack_results.csv'
    baseline_path = TABLES_DIR / 'baseline_results.csv'

    # ---- Load data for inference-based tests ----
    X_test_scaled, X_test_unscaled, y_test = load_data()
    thresholds = load_optimal_thresholds()
    preds, _ = load_models_and_predict(
        X_test_scaled, X_test_unscaled, y_test, thresholds)

    # ---- Run tests ----
    if preds:
        run_bootstrap_cis(preds, y_test)
        if len(preds) >= 2:
            run_mcnemar(preds, y_test)

    run_wilcoxon_adversarial(adv_path)
    run_friedman_attack_comparison(adv_path)
    run_effect_sizes(adv_path)
    run_degradation_summary(adv_path)
    run_robustness_correlation(adv_path, baseline_path)

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nAll tables saved to: {TABLES_DIR}")
    print("\nFor LaTeX import use pandas df.to_latex(index=False, float_format='%.4f')")
    print("\nIEEE reporting checklist:")
    print("  ✓ Table 1 — Bootstrap CIs (BCa) for all primary metrics")
    print("  ✓ Table 2 — Pairwise significance with multiple comparison correction")
    print("  ✓ Table 3 — Clean vs adversarial test with effect size")
    print("  ✓ Table 4 — Attack type comparison (Friedman + post-hoc)")
    print("  ✓ Table 5 — Effect sizes (Cohen's d, Mann-Whitney r)")
    print("  ✓ Table 6 — Degradation summary for Results section")
    print("  ✓ Table 7 — Clean accuracy / robustness correlation")


if __name__ == "__main__":
    main()
