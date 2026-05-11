"""
Visualization Functions
Generate plots and figures for model evaluation
"""

from evaluation.metrics import get_roc_data
from config.paths import FIGURES_DIR, BASELINE_FIGURES, ADVERSARIAL_FIGURES
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def plot_roc_curves(models, X_test, y_test, save_path=None):
    """
    Plot ROC curves for multiple models

    Args:
        models: Dict of {model_name: model}
        X_test: Test features
        y_test: Test labels
        save_path: Path to save figure (relative to FIGURES_DIR)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, model in models.items():
        fpr, tpr, _, auc = get_roc_data(model, X_test, y_test)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)

    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')

    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_title('ROC Curves - Model Comparison',
                 fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        full_path = FIGURES_DIR / save_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {full_path}")
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(cm, model_name, save_path=None):
    """
    Plot confusion matrix for a single model

    Args:
        cm: Confusion matrix
        model_name: Name of the model
        save_path: Full path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                cbar=False, square=True, linewidths=1, linecolor='black')

    ax.set_title(f'Confusion Matrix - {model_name}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xticklabels(['Normal', 'Attack'])
    ax.set_yticklabels(['Normal', 'Attack'])

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_confusion_matrices(models, X_test, y_test, save_dir=None, prefix=""):
    """
    Plot confusion matrices for all models

    Args:
        models: Dict of {model_name: model}
        X_test: Test features
        y_test: Test labels
        save_dir: Directory to save figures (relative to FIGURES_DIR)
    """
    from sklearn.metrics import confusion_matrix
    for name, model in models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        if save_dir:
            save_path = FIGURES_DIR / save_dir / \
                f"{prefix}confusion_{name}.png"
            plot_confusion_matrix(cm, name, save_path=save_path)
        else:
            plot_confusion_matrix(cm, name)


def plot_model_comparison(results_df, save_path=None):
    """
    Create comprehensive model comparison plots

    Args:
        results_df: DataFrame with model results
        save_path: Path to save figure (relative to FIGURES_DIR)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Sort by F1-score
    results_sorted = results_df.sort_values('f1', ascending=True)

    # 1. F1-Score comparison
    ax = axes[0, 0]
    bars = ax.barh(range(len(results_sorted)), results_sorted['f1'],
                   color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(results_sorted)))
    ax.set_yticklabels(results_sorted['model_name'])
    ax.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Model F1-Score Ranking', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1.05])
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (idx, row) in enumerate(results_sorted.iterrows()):
        ax.text(row['f1'] + 0.01, i, f"{row['f1']:.4f}",
                va='center', fontsize=10, fontweight='bold')

    # 2. Training time comparison
    ax = axes[0, 1]
    if 'training_time' in results_sorted.columns:
        results_time_sorted = results_sorted.sort_values(
            'training_time', ascending=True)
        bars = ax.barh(range(len(results_time_sorted)), results_time_sorted['training_time'],
                       color='coral', edgecolor='black')
        ax.set_yticks(range(len(results_time_sorted)))
        ax.set_yticklabels(results_time_sorted['model_name'])
        ax.set_xlabel('Training Time (seconds)',
                      fontsize=12, fontweight='bold')
        ax.set_title('Model Training Speed', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        for i, (idx, row) in enumerate(results_time_sorted.iterrows()):
            ax.text(row['training_time'] + max(results_time_sorted['training_time']) * 0.02,
                    i, f"{row['training_time']:.1f}s", va='center', fontsize=10)

    # 3. Precision vs Recall scatter
    ax = axes[1, 0]
    scatter = ax.scatter(results_df['recall'], results_df['precision'],
                         s=200, c=results_df['f1'], cmap='viridis',
                         edgecolors='black', linewidth=2, alpha=0.7)

    for idx, row in results_df.iterrows():
        ax.annotate(row['model_name'],
                    (row['recall'], row['precision']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')

    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision vs Recall Trade-off',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('F1-Score', fontsize=11, fontweight='bold')

    # 4. Performance heatmap
    ax = axes[1, 1]
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    heatmap_data = results_df.set_index('model_name')[metrics_to_plot].T

    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                ax=ax, cbar_kws={'label': 'Score'},
                vmin=0.5, vmax=1.0, linewidths=1, linecolor='black')

    ax.set_title('Performance Metrics Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metrics', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        full_path = FIGURES_DIR / save_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {full_path}")
    else:
        plt.show()

    plt.close()


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """
    Plot feature importance for tree-based models

    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
        top_n: Number of top features to show
        save_path: Path to save figure
    """
    # Extract model from pipeline if needed
    if hasattr(model, 'named_steps'):
        actual_model = model.named_steps['model']
    else:
        actual_model = model

    if not hasattr(actual_model, 'feature_importances_'):
        print(f"Model does not have feature_importances_ attribute")
        return

    # Get feature importances
    importances = actual_model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))

    # Plot
    ax.barh(range(top_n), importances[indices],
            color='steelblue', edgecolor='black')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Feature Importances',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_learning_curves(train_scores, val_scores, train_sizes, save_path=None):
    """
    Plot learning curves

    Args:
        train_scores: Training scores
        val_scores: Validation scores
        train_sizes: Training set sizes
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(train_sizes, train_scores, 'o-',
            label='Training Score', linewidth=2)
    ax.plot(train_sizes, val_scores, 'o-',
            label='Validation Score', linewidth=2)

    ax.set_xlabel('Training Set Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Learning Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


# Example usage
if __name__ == "__main__":
    print("Visualization module - use in experiments")
