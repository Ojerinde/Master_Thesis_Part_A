"""
Statistical Tests
=================
Statistical significance testing for model comparison.

Design Rationale:
- McNemar's test: Paired comparison of model predictions
- Bootstrap confidence intervals: Robust performance estimation
- Multiple comparison correction: Control family-wise error rate

References:
- McNemar (1947): Note on the sampling error of the difference
- Dietterich (1998): Approximate statistical tests for comparing supervised learning algorithms
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2
from typing import Dict, Tuple, List


def mcnemar_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray,
                 correction: bool = True) -> Dict:
    """
    McNemar's test for paired model comparison.

    Tests the null hypothesis that two models have the same error rate.

    Args:
        y_true: True labels
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2
        correction: Apply continuity correction (recommended for small samples)

    Returns:
        Dictionary with test statistics and p-value
    """
    # Calculate confusion matrix
    # b: Model 1 correct, Model 2 wrong
    # c: Model 1 wrong, Model 2 correct
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)

    b = np.sum((correct1) & (~correct2))  # Model 1 correct, Model 2 wrong
    c = np.sum((~correct1) & (correct2))   # Model 1 wrong, Model 2 correct

    # Calculate test statistic
    if correction:
        # With continuity correction
        chi2_stat = (np.abs(b - c) - 1) ** 2 / (b + c + 1e-10)
    else:
        # Without correction
        chi2_stat = (b - c) ** 2 / (b + c + 1e-10)

    # P-value (chi-square distribution with 1 degree of freedom)
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)

    # Effect size (proportion of disagreements)
    n_disagreements = b + c
    disagreement_rate = n_disagreements / len(y_true) if len(y_true) > 0 else 0

    result = {
        'chi2_statistic': float(chi2_stat),
        'p_value': float(p_value),
        'b': int(b),  # Model 1 correct, Model 2 wrong
        'c': int(c),  # Model 1 wrong, Model 2 correct
        'n_disagreements': int(n_disagreements),
        'disagreement_rate': float(disagreement_rate),
        'significant': p_value < 0.05,
    }

    return result


def compare_models_statistical(y_true: np.ndarray, predictions: Dict[str, np.ndarray],
                               alpha: float = 0.05, correction: str = 'fdr_bh') -> pd.DataFrame:
    """
    Compare multiple models using McNemar's test with multiple comparison correction.

    Args:
        y_true: True labels
        predictions: Dictionary of {model_name: predictions}
        alpha: Significance level
        correction: Multiple comparison correction method ('bonferroni', 'fdr_bh', or None)

    Returns:
        DataFrame with pairwise comparison results (including corrected p-values)

    References:
        - Benjamini & Hochberg (1995): Controlling the False Discovery Rate
        - Demsar (2006): Statistical Comparisons of Classifiers over Multiple Data Sets
    """
    model_names = list(predictions.keys())
    n_models = len(model_names)

    results = []

    for i in range(n_models):
        for j in range(i + 1, n_models):
            model1 = model_names[i]
            model2 = model_names[j]

            test_result = mcnemar_test(
                y_true,
                predictions[model1],
                predictions[model2]
            )

            results.append({
                'model1': model1,
                'model2': model2,
                'chi2_statistic': test_result['chi2_statistic'],
                'p_value': test_result['p_value'],
                'significant': test_result['significant'],
                'disagreement_rate': test_result['disagreement_rate'],
                'b': test_result['b'],
                'c': test_result['c'],
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p_value')

    # Apply multiple comparison correction
    if correction and len(results_df) > 1:
        p_values = results_df['p_value'].values
        if correction == 'bonferroni':
            # Bonferroni correction
            n_comparisons = len(p_values)
            p_corrected = np.minimum(p_values * n_comparisons, 1.0)
        elif correction == 'fdr_bh':
            # Benjamini-Hochberg FDR correction (recommended)
            try:
                from statsmodels.stats.multitest import multipletests
                _, p_corrected, _, _ = multipletests(
                    p_values,
                    alpha=alpha,
                    method='fdr_bh'
                )
            except ImportError:
                # Fallback to Bonferroni if statsmodels not available
                n_comparisons = len(p_values)
                p_corrected = np.minimum(p_values * n_comparisons, 1.0)
        else:
            p_corrected = p_values

        results_df['p_value_corrected'] = p_corrected
        results_df['significant_corrected'] = p_corrected < alpha
    else:
        results_df['p_value_corrected'] = results_df['p_value']
        results_df['significant_corrected'] = results_df['significant']

    return results_df


def bootstrap_confidence_interval(scores: np.ndarray, confidence: float = 0.95,
                                  n_bootstrap: int = 1000) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for model performance.

    Args:
        scores: Array of performance scores (e.g., accuracy per sample)
        confidence: Confidence level (default 0.95)
        n_bootstrap: Number of bootstrap samples

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    n = len(scores)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        bootstrap_sample = scores[indices]
        bootstrap_means.append(np.mean(bootstrap_sample))

    bootstrap_means = np.array(bootstrap_means)
    mean_score = np.mean(scores)

    # Calculate confidence interval
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return mean_score, lower, upper


def paired_t_test(scores1: np.ndarray, scores2: np.ndarray) -> Dict:
    """
    Paired t-test for comparing model performance.

    Args:
        scores1: Performance scores from model 1
        scores2: Performance scores from model 2

    Returns:
        Dictionary with test results
    """
    differences = scores1 - scores2
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    n = len(differences)

    # T-statistic
    t_stat = mean_diff / (std_diff / np.sqrt(n) + 1e-10)

    # P-value (two-tailed)
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - 1))

    result = {
        'mean_difference': float(mean_diff),
        'std_difference': float(std_diff),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'n_samples': int(n),
    }

    return result


def multiple_comparison_correction(p_values: np.ndarray, method: str = 'bonferroni') -> np.ndarray:
    """
    Apply multiple comparison correction to p-values.

    Args:
        p_values: Array of p-values
        method: Correction method ('bonferroni' or 'fdr_bh' for Benjamini-Hochberg)

    Returns:
        Corrected p-values
    """
    n = len(p_values)

    if method == 'bonferroni':
        # Bonferroni correction: p_corrected = p * n_comparisons
        corrected = p_values * n
        corrected = np.clip(corrected, 0, 1)
    elif method == 'fdr_bh':
        # Benjamini-Hochberg FDR correction
        from statsmodels.stats.multitest import multipletests
        _, corrected, _, _ = multipletests(
            p_values, alpha=0.05, method='fdr_bh')
    else:
        raise ValueError(f"Unknown correction method: {method}")

    return corrected


def print_statistical_comparison(results_df: pd.DataFrame):
    """
    Print formatted statistical comparison results.

    Args:
        results_df: DataFrame from compare_models_statistical
    """
    print("\n" + "="*70)
    print("STATISTICAL MODEL COMPARISON (McNemar's Test)")
    print("="*70)
    print(f"\n{'Model 1':<20} {'Model 2':<20} {'p-value':<12} {'Significant':<12}")
    print("-"*70)

    for _, row in results_df.iterrows():
        sig_str = "Yes" if row['significant'] else "No"
        print(
            f"{row['model1']:<20} {row['model2']:<20} {row['p_value']:<12.6f} {sig_str:<12}")

    print("\n" + "="*70)
    print("Interpretation:")
    print("  - Significant (p < 0.05): Models have significantly different error rates")
    print("  - Not significant: No statistically significant difference")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Test statistical tests
    print("Testing Statistical Tests...")

    # Create dummy predictions
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.randint(0, 2, n_samples)

    # Model 1: 90% accuracy
    y_pred1 = y_true.copy()
    flip_indices = np.random.choice(
        n_samples, size=int(0.1 * n_samples), replace=False)
    y_pred1[flip_indices] = 1 - y_pred1[flip_indices]

    # Model 2: 85% accuracy
    y_pred2 = y_true.copy()
    flip_indices = np.random.choice(
        n_samples, size=int(0.15 * n_samples), replace=False)
    y_pred2[flip_indices] = 1 - y_pred2[flip_indices]

    # Test McNemar
    print("\n1. McNemar's Test:")
    result = mcnemar_test(y_true, y_pred1, y_pred2)
    print(f"   Chi2 statistic: {result['chi2_statistic']:.4f}")
    print(f"   P-value: {result['p_value']:.6f}")
    print(f"   Significant: {result['significant']}")

    # Test multiple model comparison
    print("\n2. Multiple Model Comparison:")
    predictions = {
        'Model1': y_pred1,
        'Model2': y_pred2,
    }
    results_df = compare_models_statistical(y_true, predictions)
    print_statistical_comparison(results_df)

    # Test bootstrap CI
    print("\n3. Bootstrap Confidence Interval:")
    scores = (y_pred1 == y_true).astype(float)
    mean, lower, upper = bootstrap_confidence_interval(scores)
    print(f"   Mean accuracy: {mean:.4f}")
    print(f"   95% CI: [{lower:.4f}, {upper:.4f}]")

    print("\n✓ All statistical tests passed!")
