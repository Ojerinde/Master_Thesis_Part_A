"""
Model Evaluation Metrics
Comprehensive evaluation functions for ML models with GNSS-specific metrics.

References:
- GNSS spoofing detection evaluation metrics
- Receiver Operating Characteristic (ROC) analysis
- Precision-Recall curves for imbalanced datasets
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    average_precision_score,
)


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Comprehensive evaluation of a trained model

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model

    Returns:
        dict: Evaluation metrics
    """
    # Verify input shapes
    if X_test.shape[0] != len(y_test):
        raise ValueError(
            f"Shape mismatch in evaluate_model for {model_name}: "
            f"X_test has {X_test.shape[0]} samples but y_test has {len(y_test)} samples"
        )
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Verify prediction shape
    if len(y_pred) != len(y_test):
        raise ValueError(
            f"Prediction shape mismatch for {model_name}: "
            f"y_pred has {len(y_pred)} samples but y_test has {len(y_test)} samples. "
            f"X_test shape: {X_test.shape}"
        )

    # Probabilities (if available)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = y_pred  # Fallback

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_proba) if len(np.unique(y_proba)) > 1 else 0.5,
        'avg_precision': average_precision_score(y_test, y_proba) if len(np.unique(y_proba)) > 1 else 0.5,
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm

    # Detailed per-class metrics
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)

        # Specificity and other metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        # Negative Predictive Value
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        # False Positive Rate
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        # False Negative Rate
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0

    return metrics


def compare_models(results_list, sort_by='f1'):
    """
    Compare multiple models

    Args:
        results_list: List of metric dictionaries from evaluate_model
        sort_by: Metric to sort by

    Returns:
        DataFrame: Comparison table
    """
    df = pd.DataFrame(results_list)

    # Select key metrics for comparison
    key_metrics = ['model_name', 'accuracy',
                   'precision', 'recall', 'f1', 'auc_roc']
    if 'training_time' in df.columns:
        key_metrics.append('training_time')

    df_compare = df[key_metrics].copy()
    df_compare = df_compare.sort_values(sort_by, ascending=False)

    return df_compare


def print_classification_report(model, X_test, y_test, model_name="Model"):
    """
    Print detailed classification report

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
    """
    y_pred = model.predict(X_test)

    print(f"\n{'='*70}")
    print(f"Classification Report: {model_name}")
    print(f"{'='*70}")
    print(classification_report(y_test, y_pred,
          target_names=['Normal', 'Attack']))


def calculate_attack_detection_rate(model, X_test, y_test):
    """
    Calculate attack detection rate (recall for attack class)

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels

    Returns:
        float: Attack detection rate
    """
    y_pred = model.predict(X_test)
    attack_mask = y_test == 1

    if attack_mask.sum() == 0:
        return 0.0

    attack_detected = (y_pred[attack_mask] == 1).sum()
    detection_rate = attack_detected / attack_mask.sum()

    return detection_rate


def calculate_false_alarm_rate(model, X_test, y_test):
    """
    Calculate false alarm rate (FPR)

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels

    Returns:
        float: False alarm rate
    """
    y_pred = model.predict(X_test)
    normal_mask = y_test == 0

    if normal_mask.sum() == 0:
        return 0.0

    false_alarms = (y_pred[normal_mask] == 1).sum()
    false_alarm_rate = false_alarms / normal_mask.sum()

    return false_alarm_rate


def get_roc_data(model, X_test, y_test):
    """
    Get ROC curve data

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels

    Returns:
        tuple: (fpr, tpr, thresholds, auc)
    """
    # Get probabilities
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = model.predict(X_test)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba) if len(
        np.unique(y_proba)) > 1 else 0.5

    return fpr, tpr, thresholds, auc


def evaluate_gnss_spoofing_detection(model, X_test, y_test, 
                                     feature_names=None, verbose=True):
    """
    GNSS-specific evaluation metrics for spoofing detection.
    
    GNSS spoofing detection requires:
    - High attack detection rate (low false negatives)
    - Low false alarm rate (high specificity)
    - Fast detection (latency considerations)
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels (0=normal, 1=attack)
        feature_names: Optional feature names for analysis
        verbose: Print detailed results
    
    Returns:
        dict: GNSS-specific metrics
    
    References:
        - Psiaki & Humphreys (2016): GNSS Spoofing and Detection
        - Wesson et al. (2011): TEXBAT Dataset
    """
    from evaluation.metrics import evaluate_model
    
    # Standard metrics
    standard_metrics = evaluate_model(model, X_test, y_test)
    
    # GNSS-specific metrics
    gnss_metrics = {
        # Attack detection metrics
        'attack_detection_rate': calculate_attack_detection_rate(model, X_test, y_test),
        'false_alarm_rate': calculate_false_alarm_rate(model, X_test, y_test),
        
        # Security metrics
        'missed_detection_rate': 1.0 - calculate_attack_detection_rate(model, X_test, y_test),
        'true_negative_rate': standard_metrics.get('specificity', 0.0),
        
        # Combined security score (weighted by importance)
        # High weight on attack detection (critical for security)
        'security_score': (
            0.7 * calculate_attack_detection_rate(model, X_test, y_test) +
            0.3 * standard_metrics.get('specificity', 0.0)
        ),
        
        # Standard metrics included
        **standard_metrics
    }
    
    if verbose:
        print("\n" + "="*70)
        print("GNSS Spoofing Detection Metrics")
        print("="*70)
        print(f"Attack Detection Rate:     {gnss_metrics['attack_detection_rate']:.4f}")
        print(f"False Alarm Rate:          {gnss_metrics['false_alarm_rate']:.4f}")
        print(f"Missed Detection Rate:    {gnss_metrics['missed_detection_rate']:.4f}")
        print(f"True Negative Rate:       {gnss_metrics['true_negative_rate']:.4f}")
        print(f"Security Score:            {gnss_metrics['security_score']:.4f}")
        print(f"F1-Score:                  {gnss_metrics['f1']:.4f}")
        print(f"AUC-ROC:                   {gnss_metrics['auc_roc']:.4f}")
        print("="*70)
    
    return gnss_metrics


def calculate_detection_latency(model, X_test, y_test, time_window=1.0):
    """
    Calculate average detection latency for spoofing attacks.
    
    Note: This is a placeholder for time-series data.
    For actual latency calculation, requires temporal data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        time_window: Time window for detection (seconds)
    
    Returns:
        dict: Latency metrics
    """
    # Placeholder implementation
    # In real scenario, would track time from attack start to detection
    y_pred = model.predict(X_test)
    attack_mask = y_test == 1
    
    if attack_mask.sum() == 0:
        return {
            'avg_latency': 0.0,
            'median_latency': 0.0,
            'p95_latency': 0.0
        }
    
    # Simplified: assume detection happens within time_window
    # Real implementation would track actual timestamps
    detected_attacks = (y_pred[attack_mask] == 1).sum()
    detection_rate = detected_attacks / attack_mask.sum()
    
    # Estimate latency (simplified)
    avg_latency = time_window * (1 - detection_rate)  # Undetected attacks have full latency
    
    return {
        'avg_latency': avg_latency,
        'median_latency': avg_latency,
        'p95_latency': time_window,
        'detection_rate': detection_rate
    }


# Example usage
if __name__ == "__main__":
    print("Metrics module - use in experiments")
