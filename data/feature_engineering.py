"""
GNSS Feature Engineering — all statistics derived from training set only
to prevent data leakage.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import StandardScaler


class SafeFeatureEngineer:
    """
    Feature engineering that prevents data leakage by using only training statistics.

    All transformations are fitted on training data and then applied to test data.
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.feature_stats = {}  # Store training statistics
        self.feature_names = None

    def fit_transform(self, X_train: np.ndarray, X_test: np.ndarray,
                      feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Fit feature engineering on training data and transform both sets.

        Args:
            X_train: Training features (n_samples_train, n_features)
            X_test: Test features (n_samples_test, n_features)
            feature_names: Optional list of feature names

        Returns:
            Tuple of (X_train_engineered, X_test_engineered, new_feature_names)
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        # Store original feature names (before engineering)
        self.original_feature_names = feature_names.copy()
        # Will be updated to include engineered features
        self.feature_names = feature_names

        # Convert to DataFrame for easier manipulation
        df_train = pd.DataFrame(X_train, columns=feature_names)
        df_test = pd.DataFrame(X_test, columns=feature_names)

        # Identify GNSS feature types from training data
        doppler_cols = [
            col for col in feature_names if 'doppler' in col.lower()]
        cn0_cols = [col for col in feature_names if 'cn0' in col.lower(
        ) or 'c/n0' in col.lower() or 'c_n0' in col.lower()]
        pr_cols = [col for col in feature_names if 'pseudorange' in col.lower() or (
            'pr' in col.lower() and 'ratio' not in col.lower())]

        new_features_train = []
        new_features_test = []
        new_feature_names = list(feature_names)  # Start with original features

        # 1. Normalized features (using training min/max)
        for col in doppler_cols + cn0_cols:
            if col in df_train.columns:
                col_min = df_train[col].min()
                col_max = df_train[col].max()
                col_range = col_max - col_min

                # Store statistics
                self.feature_stats[f'{col}_norm'] = {
                    'min': col_min, 'max': col_max, 'range': col_range}

                # Transform training (normalize to [0, 1])
                if col_range > 1e-10:
                    train_norm = (df_train[col] - col_min) / col_range
                    test_norm = (df_test[col] - col_min) / \
                        col_range  # Use training min/max!
                else:
                    train_norm = pd.Series(0.5, index=df_train.index)
                    test_norm = pd.Series(0.5, index=df_test.index)

                new_features_train.append(train_norm.values)
                new_features_test.append(test_norm.values)
                new_feature_names.append(f'{col}_normalized')

                if self.verbose:
                    print(
                        f"  Created {col}_normalized (train range: [{col_min:.2f}, {col_max:.2f}])")

        # 2. Interaction features (using training statistics)
        if doppler_cols and cn0_cols:
            doppler_col = doppler_cols[0]
            cn0_col = cn0_cols[0]

            if doppler_col in df_train.columns and cn0_col in df_train.columns:
                # Doppler/CN0 ratio
                train_ratio = np.abs(
                    df_train[doppler_col]) / (df_train[cn0_col] + 1e-10)
                test_ratio = np.abs(
                    df_test[doppler_col]) / (df_test[cn0_col] + 1e-10)

                new_features_train.append(train_ratio.values)
                new_features_test.append(test_ratio.values)
                new_feature_names.append('doppler_cn0_ratio')

                if self.verbose:
                    print(f"  Created doppler_cn0_ratio")

        if pr_cols and cn0_cols:
            pr_col = pr_cols[0]
            cn0_col = cn0_cols[0]

            if pr_col in df_train.columns and cn0_col in df_train.columns:
                # Pseudorange/CN0 ratio
                train_ratio = df_train[pr_col] / (df_train[cn0_col] + 1e-10)
                test_ratio = df_test[pr_col] / (df_test[cn0_col] + 1e-10)

                new_features_train.append(train_ratio.values)
                new_features_test.append(test_ratio.values)
                new_feature_names.append('pseudorange_cn0_ratio')

                if self.verbose:
                    print(f"  Created pseudorange_cn0_ratio")

        # 3. Deviation features (Z-scores using training mean/std)
        for col in doppler_cols + cn0_cols:
            if col in df_train.columns:
                mean_val = df_train[col].mean()
                std_val = df_train[col].std()

                # Store statistics
                self.feature_stats[f'{col}_dev'] = {
                    'mean': mean_val, 'std': std_val}

                # Transform using training statistics
                if std_val > 1e-10:
                    train_dev = (df_train[col] - mean_val) / std_val
                    test_dev = (df_test[col] - mean_val) / \
                        std_val  # Use training mean/std!
                else:
                    train_dev = pd.Series(0.0, index=df_train.index)
                    test_dev = pd.Series(0.0, index=df_test.index)

                new_features_train.append(train_dev.values)
                new_features_test.append(test_dev.values)
                new_feature_names.append(f'{col}_deviation')

                if self.verbose:
                    print(
                        f"  Created {col}_deviation (train mean: {mean_val:.2f}, std: {std_val:.2f})")

        # 4. Binary flags (using training thresholds)
        for col in doppler_cols:
            if col in df_train.columns:
                mean_val = df_train[col].mean()
                std_val = df_train[col].std()

                # Store statistics
                self.feature_stats[f'{col}_extreme'] = {
                    'mean': mean_val, 'std': std_val}

                # Transform using training thresholds
                if std_val > 1e-10:
                    threshold = 3 * std_val
                    train_flag = (
                        np.abs(df_train[col] - mean_val) > threshold).astype(int)
                    # Use training threshold!
                    test_flag = (
                        np.abs(df_test[col] - mean_val) > threshold).astype(int)
                else:
                    train_flag = pd.Series(0, index=df_train.index)
                    test_flag = pd.Series(0, index=df_test.index)

                new_features_train.append(train_flag.values)
                new_features_test.append(test_flag.values)
                new_feature_names.append(f'is_extreme_{col}')

                if self.verbose:
                    print(
                        f"  Created is_extreme_{col} (threshold: {mean_val:.2f} ± {3*std_val:.2f})")

        for col in cn0_cols:
            if col in df_train.columns:
                # Low CN0 threshold (bottom quartile from training)
                threshold = df_train[col].quantile(0.25)

                # Store statistics
                self.feature_stats[f'{col}_low'] = {'threshold': threshold}

                # Transform using training threshold
                train_flag = (df_train[col] < threshold).astype(int)
                test_flag = (df_test[col] < threshold).astype(
                    int)  # Use training threshold!

                new_features_train.append(train_flag.values)
                new_features_test.append(test_flag.values)
                new_feature_names.append(f'is_low_{col}')

                if self.verbose:
                    print(
                        f"  Created is_low_{col} (threshold: {threshold:.2f})")

        # 5. Signal quality index (using training normalization)
        if doppler_cols and cn0_cols:
            doppler_col = doppler_cols[0]
            cn0_col = cn0_cols[0]

            if doppler_col in df_train.columns and cn0_col in df_train.columns:
                # Normalize using training statistics
                doppler_min = df_train[doppler_col].min()
                doppler_max = df_train[doppler_col].max()
                doppler_range = doppler_max - doppler_min

                cn0_min = df_train[cn0_col].min()
                cn0_max = df_train[cn0_col].max()
                cn0_range = cn0_max - cn0_min

                if doppler_range > 1e-10 and cn0_range > 1e-10:
                    # Training
                    doppler_norm_train = (
                        df_train[doppler_col] - doppler_min) / doppler_range
                    cn0_norm_train = (df_train[cn0_col] - cn0_min) / cn0_range
                    train_sqi = cn0_norm_train / \
                        (np.abs(doppler_norm_train) + 1e-10)

                    # Test (using training normalization)
                    doppler_norm_test = (
                        df_test[doppler_col] - doppler_min) / doppler_range
                    cn0_norm_test = (df_test[cn0_col] - cn0_min) / cn0_range
                    test_sqi = cn0_norm_test / \
                        (np.abs(doppler_norm_test) + 1e-10)

                    new_features_train.append(train_sqi.values)
                    new_features_test.append(test_sqi.values)
                    new_feature_names.append('signal_quality_index')

                    if self.verbose:
                        print(f"  Created signal_quality_index")

        # Combine all features
        if new_features_train:
            X_train_engineered = np.hstack(
                [X_train] + [f.reshape(-1, 1) if f.ndim == 1 else f for f in new_features_train])
            X_test_engineered = np.hstack(
                [X_test] + [f.reshape(-1, 1) if f.ndim == 1 else f for f in new_features_test])
        else:
            X_train_engineered = X_train
            X_test_engineered = X_test

        # Update feature_names to include engineered features (for reference)
        self.feature_names = new_feature_names

        if self.verbose:
            print(f"\nFeature engineering complete:")
            print(f"  Original features: {len(self.original_feature_names)}")
            print(f"  Engineered features: {len(new_feature_names)}")
            print(f"  Total features: {X_train_engineered.shape[1]}")

        return X_train_engineered, X_test_engineered, new_feature_names

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data using already-fitted statistics.

        Args:
            X: Features to transform (n_samples, n_features) - must match ORIGINAL feature count

        Returns:
            Transformed features with same number of engineered features as fit_transform
        """
        if not self.feature_stats:
            raise ValueError("Must call fit_transform first before transform")

        if self.feature_names is None:
            raise ValueError("Feature names not set")

        # Get original feature names (stored during fit_transform)
        original_feature_names = getattr(self, 'original_feature_names', None)
        if original_feature_names is None:
            # Fallback: extract from feature_names (original features are first N)
            # Filter out engineered feature names
            base_names = [name for name in self.feature_names
                          if not any(suffix in name for suffix in ['_normalized', '_deviation', '_ratio', 'is_extreme_', 'is_low_', 'signal_quality'])]
            original_feature_names = base_names[:X.shape[1]] if len(
                base_names) >= X.shape[1] else self.feature_names[:X.shape[1]]

        # Verify input shape matches original feature count
        if X.shape[1] != len(original_feature_names):
            raise ValueError(
                f"Input has {X.shape[1]} features but expected {len(original_feature_names)} original features. "
                f"Original feature names: {original_feature_names[:5]}..."
            )

        # Convert to DataFrame using original feature names
        df = pd.DataFrame(X, columns=original_feature_names)

        new_features = []

        # Identify GNSS feature types (use original feature names)
        doppler_cols = [
            col for col in original_feature_names if 'doppler' in col.lower()]
        cn0_cols = [col for col in original_feature_names if 'cn0' in col.lower(
        ) or 'c/n0' in col.lower() or 'c_n0' in col.lower()]
        pr_cols = [col for col in original_feature_names if 'pseudorange' in col.lower(
        ) or ('pr' in col.lower() and 'ratio' not in col.lower())]

        # Apply same transformations using stored statistics
        # 1. Normalized features
        for col in doppler_cols + cn0_cols:
            if col in df.columns and f'{col}_norm' in self.feature_stats:
                stats = self.feature_stats[f'{col}_norm']
                col_min = stats['min']
                col_range = stats['range']

                if col_range > 1e-10:
                    norm = (df[col] - col_min) / col_range
                else:
                    norm = pd.Series(0.5, index=df.index)

                new_features.append(norm.values.reshape(-1, 1))

        # 2. Interaction features
        if doppler_cols and cn0_cols:
            doppler_col = doppler_cols[0]
            cn0_col = cn0_cols[0]
            if doppler_col in df.columns and cn0_col in df.columns:
                ratio = np.abs(df[doppler_col]) / (df[cn0_col] + 1e-10)
                new_features.append(ratio.values.reshape(-1, 1))

        if pr_cols and cn0_cols:
            pr_col = pr_cols[0]
            cn0_col = cn0_cols[0]
            if pr_col in df.columns and cn0_col in df.columns:
                ratio = df[pr_col] / (df[cn0_col] + 1e-10)
                new_features.append(ratio.values.reshape(-1, 1))

        # 3. Deviation features
        for col in doppler_cols + cn0_cols:
            if col in df.columns and f'{col}_dev' in self.feature_stats:
                stats = self.feature_stats[f'{col}_dev']
                mean_val = stats['mean']
                std_val = stats['std']

                if std_val > 1e-10:
                    dev = (df[col] - mean_val) / std_val
                else:
                    dev = pd.Series(0.0, index=df.index)

                new_features.append(dev.values.reshape(-1, 1))

        # 4. Binary flags
        for col in doppler_cols:
            if col in df.columns and f'{col}_extreme' in self.feature_stats:
                stats = self.feature_stats[f'{col}_extreme']
                mean_val = stats['mean']
                std_val = stats['std']

                if std_val > 1e-10:
                    threshold = 3 * std_val
                    flag = (np.abs(df[col] - mean_val) > threshold).astype(int)
                else:
                    flag = pd.Series(0, index=df.index)

                new_features.append(flag.values.reshape(-1, 1))

        for col in cn0_cols:
            if col in df.columns and f'{col}_low' in self.feature_stats:
                stats = self.feature_stats[f'{col}_low']
                threshold = stats['threshold']
                flag = (df[col] < threshold).astype(int)
                new_features.append(flag.values.reshape(-1, 1))

        # 5. Signal quality index (using stored normalization stats)
        if doppler_cols and cn0_cols:
            doppler_col = doppler_cols[0]
            cn0_col = cn0_cols[0]
            if doppler_col in df.columns and cn0_col in df.columns:
                # Use stored normalization stats
                if f'{doppler_col}_norm' in self.feature_stats and f'{cn0_col}_norm' in self.feature_stats:
                    doppler_stats = self.feature_stats[f'{doppler_col}_norm']
                    cn0_stats = self.feature_stats[f'{cn0_col}_norm']

                    doppler_min = doppler_stats['min']
                    doppler_range = doppler_stats['range']
                    cn0_min = cn0_stats['min']
                    cn0_range = cn0_stats['range']

                    if doppler_range > 1e-10 and cn0_range > 1e-10:
                        doppler_norm = (df[doppler_col] -
                                        doppler_min) / doppler_range
                        cn0_norm = (df[cn0_col] - cn0_min) / cn0_range
                        sqi = cn0_norm / (np.abs(doppler_norm) + 1e-10)
                        new_features.append(sqi.values.reshape(-1, 1))

        # Combine all features
        if new_features:
            # Ensure all features are 2D arrays
            features_to_stack = [X]
            for feat in new_features:
                if feat.ndim == 1:
                    features_to_stack.append(feat.reshape(-1, 1))
                else:
                    features_to_stack.append(feat)
            X_engineered = np.hstack(features_to_stack)
        else:
            X_engineered = X

        # Verify output shape
        if X_engineered.shape[0] != X.shape[0]:
            raise ValueError(
                f"Feature engineering changed sample count: {X.shape[0]} -> {X_engineered.shape[0]}. "
                f"Input shape: {X.shape}, Output shape: {X_engineered.shape}"
            )

        return X_engineered


def safe_feature_engineering(X_train: np.ndarray, X_test: np.ndarray,
                             feature_names: Optional[List[str]] = None,
                             verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Convenience function for safe feature engineering.

    Args:
        X_train: Training features
        X_test: Test features
        feature_names: Optional feature names
        verbose: Print progress

    Returns:
        Tuple of (X_train_engineered, X_test_engineered, new_feature_names)
    """
    engineer = SafeFeatureEngineer(verbose=verbose)
    return engineer.fit_transform(X_train, X_test, feature_names)


if __name__ == "__main__":
    # Test feature engineering
    print("Testing Safe Feature Engineering...")

    np.random.seed(42)
    n_train, n_test = 1000, 200
    n_features = 5

    # Create dummy data
    X_train = np.random.randn(n_train, n_features)
    X_test = np.random.randn(n_test, n_features)

    feature_names = ['doppler_hz', 'cn0_dbhz',
                     'pseudorange', 'feature_3', 'feature_4']

    X_train_eng, X_test_eng, new_names = safe_feature_engineering(
        X_train, X_test, feature_names, verbose=True
    )

    print(f"\nOriginal shape: {X_train.shape} -> {X_test.shape}")
    print(f"Engineered shape: {X_train_eng.shape} -> {X_test_eng.shape}")
    print(f"New feature names: {len(new_names)}")
    print("\n✓ Safe feature engineering test complete!")
