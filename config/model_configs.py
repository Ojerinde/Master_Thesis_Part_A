"""
Model Configurations
====================
Hyperparameters for all classical and deep learning models.

Design Rationale:
- Classical models: Standard configurations proven effective in GNSS research
- Deep learning: Architecture choices based on signal processing literature
- All hyperparameters documented with justification for paper defense

References:
- Rustamov & Closas (2023): ML for GNSS Interference Detection
- Shafiee et al. (2024): Deep Learning for GNSS Security
- Chen & Guestrin (2016): XGBoost: A Scalable Tree Boosting System
- Ke et al. (2017): LightGBM: A Highly Efficient Gradient Boosting Decision Tree
"""

# --- Classical models (scikit-learn) ---

# Support Vector Machine (Linear)
# Rationale: Linear decision boundary included for literature comparison
# Note: Limited discriminative capability across heterogeneous attack scenarios
# (ds2, ds3, ds7 are not linearly separable), included as weak baseline
SVM_LINEAR_CONFIG = {
    'C': 0.1,
    'kernel': 'linear',
    'max_iter': 20000,
    'random_state': 42,
    'class_weight': 'balanced',
    'probability': True,
    'tol': 1e-3,
}

LINEAR_SVC_CONFIG = {
    'C': 0.1,
    'max_iter': 20000,
    'random_state': 42,
    'class_weight': 'balanced',
    'tol': 1e-4,
}

# Support Vector Machine (RBF kernel)
# Rationale: the direct comparator paper (An et al. 2025, "Adversarial Evasion
# Attacks on SVM-Based GPS Spoofing Detection Systems," Sensors 25(19):6062)
# attacks an SVM; a linear kernel already failed on this corpus for the same
# reason LogisticRegression did (not linearly separable), so this must be a
# non-linear kernel to be a meaningful comparator, not a repeat of that result.
SVM_RBF_CONFIG = {
    'C': 1.0,
    'kernel': 'rbf',
    'gamma': 'scale',
    'class_weight': 'balanced',
    'probability': True,
    'random_state': 42,
    'tol': 1e-3,
    'max_iter': 50000,
}

# Random Forest
# Rationale: Ensemble method, robust to outliers
# Widely used in GNSS research (Rustamov & Closas, 2023)
RANDOM_FOREST_CONFIG = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'random_state': 42,
    'class_weight': 'balanced',
    'n_jobs': -1,
    'max_features': 'sqrt',
}

# Gradient Boosting (sklearn)
# Rationale: Standard sklearn baseline for comparison with XGBoost/LightGBM
GRADIENT_BOOSTING_CONFIG = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 5,
    'random_state': 42,
    'subsample': 0.8,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
}

# XGBoost
# Rationale: Optimized gradient boosting with L1/L2 regularization
# Faster and more regularized than sklearn GradientBoosting
# Widely cited in recent GNSS spoofing detection literature
XGBOOST_CONFIG = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,           # L1 regularization
    'reg_lambda': 1.0,          # L2 regularization
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'min_child_weight': 5,
    'gamma': 0.1,
}

# LightGBM
# Rationale: Leaf-wise tree growth, fastest gradient boosting implementation
# Excellent performance on large GNSS datasets, increasing adoption in literature
# (Ke et al., 2017)
LIGHTGBM_CONFIG = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 6,
    'num_leaves': 31,           # Controls model complexity
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'class_weight': 'balanced',
    'verbose': -1,
    'min_child_samples': 5,            # Suppress LightGBM output
}

# Logistic Regression
# Rationale: Simple interpretable baseline, fast training
# Included for feature importance analysis and as linear reference point
LOGISTIC_REGRESSION_CONFIG = {
    'C': 0.1,
    'max_iter': 1000,
    'random_state': 42,
    'class_weight': 'balanced',
    'solver': 'lbfgs',
    'tol': 1e-3,
}

# K-Nearest Neighbors
# Rationale: Instance-based learning, no training phase
# Useful for understanding local decision boundaries in feature space
KNN_CONFIG = {
    'n_neighbors': 11,
    'weights': 'distance',
    'metric': 'minkowski',
    'n_jobs': -1,
}

# Decision Tree
# Rationale: Interpretable, visualizable decision rules
# Good for understanding which signal features drive attack detection
DECISION_TREE_CONFIG = {
    'max_depth': 20,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'random_state': 42,
    'class_weight': 'balanced',
}

# --- Deep learning models (PyTorch) ---

# 1D CNN
# Rationale: Spatial feature extraction from GNSS signal sequences
# Convolutional filters learn local patterns in signal features
CNN_1D_CONFIG = {
    'input_dim': None,
    'conv_layers': [
        {'filters': 64, 'kernel_size': 3, 'activation': 'relu'},
        {'filters': 128, 'kernel_size': 3, 'activation': 'relu'},
        {'filters': 64, 'kernel_size': 3, 'activation': 'relu'},
    ],
    'dense_layers': [128, 64],
    'dropout_rate': 0.3,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'patience': 10,
}

# LSTM
# Rationale: Temporal dependency modeling in GNSS signals
# Captures sequential patterns in signal evolution over time
LSTM_CONFIG = {
    'input_dim': None,
    'sequence_length': 1,
    'lstm_layers': [
        {'units': 128, 'return_sequences': True},
        {'units': 64, 'return_sequences': False},
    ],
    'dense_layers': [64, 32],
    'dropout_rate': 0.3,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'patience': 10,
}

# Bidirectional LSTM
# Rationale: Bidirectional context for signal analysis
# Learns from both past and future signal patterns
BILSTM_CONFIG = {
    'input_dim': None,
    'sequence_length': 1,
    'bilstm_layers': [
        {'units': 128, 'return_sequences': True},
        {'units': 64, 'return_sequences': False},
    ],
    'dense_layers': [64, 32],
    'dropout_rate': 0.3,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'patience': 10,
}

# CNN-LSTM Hybrid
# Rationale: Combined spatial-temporal learning
# CNN extracts local features, LSTM models temporal dependencies
CNN_LSTM_CONFIG = {
    'input_dim': None,
    'sequence_length': 1,
    'conv_layers': [
        {'filters': 64, 'kernel_size': 3, 'activation': 'relu'},
        {'filters': 32, 'kernel_size': 3, 'activation': 'relu'},
    ],
    'lstm_units': 64,
    'dense_layers': [32],
    'dropout_rate': 0.3,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'patience': 10,
}

# Transformer with Multi-Head Attention
# Rationale: Self-attention mechanism for GNSS signal patterns
# State-of-the-art for sequence modeling (Dosovitskiy et al., 2021)
TRANSFORMER_CONFIG = {
    'input_dim': None,
    'num_heads': 4,
    'ff_dim': 128,
    'num_transformer_blocks': 2,
    'mlp_units': [128, 64],
    'dropout_rate': 0.1,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'patience': 10,
}

# Temporal Convolutional Network (TCN)
# Rationale: Dilated convolutions capture long-range temporal dependencies
# More efficient than RNNs for sequence tasks (Bai et al., 2018)
TCN_CONFIG = {
    'input_dim': None,
    'sequence_length': 1,
    'num_filters': 64,
    'kernel_size': 3,
    'num_blocks': 4,
    'dropout_rate': 0.2,
    'dense_layers': [128, 64],
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'patience': 10,
}

# --- Training configuration ---

TRAINING_CONFIG = {
    'test_size': 0.2,
    'val_size': 0.1,
    'random_state': 42,
    'stratify': True,
    'cv_folds': 5,
    'verbose': 1,
}

# --- Helper function ---


def get_config(model_name):
    """Get configuration for a specific model."""
    config_map = {
        'svm_linear':          SVM_LINEAR_CONFIG,
        'svm_rbf':             SVM_RBF_CONFIG,
        "linearsvc":           LINEAR_SVC_CONFIG,
        'random_forest':       RANDOM_FOREST_CONFIG,
        'gradient_boosting':   GRADIENT_BOOSTING_CONFIG,
        'xgboost':             XGBOOST_CONFIG,
        'lightgbm':            LIGHTGBM_CONFIG,
        'logistic_regression': LOGISTIC_REGRESSION_CONFIG,
        'knn':                 KNN_CONFIG,
        'decision_tree':       DECISION_TREE_CONFIG,
        'cnn_1d':              CNN_1D_CONFIG,
        'lstm':                LSTM_CONFIG,
        'bilstm':              BILSTM_CONFIG,
        'cnn_lstm':            CNN_LSTM_CONFIG,
        'transformer':         TRANSFORMER_CONFIG,
        'tcn':                 TCN_CONFIG,
    }

    if model_name.lower() not in config_map:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(config_map.keys())}"
        )

    return config_map[model_name.lower()].copy()


if __name__ == "__main__":
    print("Available model configurations:")
    for name in get_config.__code__.co_consts:
        pass
    models = [
        'svm_linear', 'random_forest', 'gradient_boosting',
        'xgboost', 'lightgbm', 'logistic_regression',
        'knn', 'decision_tree', 'cnn_1d', 'lstm',
        'bilstm', 'cnn_lstm', 'transformer', 'tcn'
    ]
    for model in models:
        config = get_config(model)
        print(f"\n{model.upper()}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
