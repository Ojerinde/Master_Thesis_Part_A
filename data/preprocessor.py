"""
GNSS Data Preprocessor — scaling, splitting, and caching.
"""

from config.paths import (
    PROCESSED_DATA_DIR, TRAIN_DATA_PATH, TEST_DATA_PATH,
    VAL_DATA_PATH, FEATURE_STATS_PATH, SCALER_PATH
)
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict


class GNSSPreprocessor:
    """
    Preprocessing pipeline for GNSS adversarial research.

    Steps:
    1. Feature scaling (StandardScaler)
    2. Train/val/test split (stratified)
    3. Feature statistics computation
    4. Caching for reproducibility
    """

    def __init__(self, test_size=0.2, val_size=0.1, random_state=42, verbose=True):
        """
        Initialize preprocessor.

        Args:
            test_size: Proportion for test set (default 0.2 = 20%)
            val_size: Proportion of training set for validation (default 0.1)
            random_state: Random seed for reproducibility
            verbose: Print processing information
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.verbose = verbose
        self.scaler = StandardScaler()

    def split_data(self, features, labels):
        """Stratified train/val/test split."""
        if self.verbose:
            print("\n" + "="*70)
            print("Splitting Data")
            print("="*70 + "\n")

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels
        )

        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=y_temp
        )

        if self.verbose:
            print(
                f"Training set:   {len(X_train):,} samples ({len(X_train)/len(features)*100:.1f}%)")
            print(
                f"Validation set: {len(X_val):,} samples ({len(X_val)/len(features)*100:.1f}%)")
            print(
                f"Test set:       {len(X_test):,} samples ({len(X_test)/len(features)*100:.1f}%)")

            # Check class balance
            print("\nClass distribution:")
            for set_name, y_set in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
                unique, counts = np.unique(y_set, return_counts=True)
                dist_str = ", ".join([f"Class {u}: {c} ({c/len(y_set)*100:.1f}%)"
                                     for u, c in zip(unique, counts)])
                print(f"  {set_name}: {dist_str}")

        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
        }

    def scale_features(self, data_dict):
        """Fit StandardScaler on train only, transform all sets."""
        if self.verbose:
            print("\n" + "="*70)
            print("Scaling Features")
            print("="*70 + "\n")

        # Fit scaler on training data only
        self.scaler.fit(data_dict['X_train'])

        # Transform all sets
        data_dict['X_train_scaled'] = self.scaler.transform(
            data_dict['X_train'])
        data_dict['X_val_scaled'] = self.scaler.transform(data_dict['X_val'])
        data_dict['X_test_scaled'] = self.scaler.transform(data_dict['X_test'])

        if self.verbose:
            print("Scaler fitted on training data")
            print(f"Feature mean: {self.scaler.mean_[:5]} ... (first 5)")
            print(f"Feature std:  {self.scaler.scale_[:5]} ... (first 5)")
            print("All sets transformed")

        return data_dict

    def compute_feature_statistics(self, features, labels, feature_names=None):
        """Compute per-feature statistics including class-wise stats."""
        if self.verbose:
            print("\n" + "="*70)
            print("Computing Feature Statistics")
            print("="*70 + "\n")

        stats = {
            'num_features': features.shape[1],
            'num_samples': features.shape[0],
            'feature_stats': []
        }

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(features.shape[1])]

        for i, name in enumerate(feature_names):
            feat_values = features[:, i]

            feat_stat = {
                'name': name,
                'index': i,
                'min': float(np.min(feat_values)),
                'max': float(np.max(feat_values)),
                'mean': float(np.mean(feat_values)),
                'std': float(np.std(feat_values)),
                'median': float(np.median(feat_values)),
                'q25': float(np.percentile(feat_values, 25)),
                'q75': float(np.percentile(feat_values, 75)),
            }

            # Class-wise statistics
            for label in np.unique(labels):
                mask = labels == label
                feat_stat[f'mean_class_{label}'] = float(
                    np.mean(feat_values[mask]))
                feat_stat[f'std_class_{label}'] = float(
                    np.std(feat_values[mask]))

            stats['feature_stats'].append(feat_stat)

        if self.verbose:
            print(f"Computed statistics for {len(feature_names)} features")

        return stats

    def save_processed_data(self, data_dict, feature_stats):
        """Save processed data and statistics to disk."""
        if self.verbose:
            print("\n" + "="*70)
            print("Saving Processed Data")
            print("="*70 + "\n")

        # Ensure directory exists
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Save train/val/test sets
        train_data = {
            'X': data_dict['X_train_scaled'],
            'y': data_dict['y_train'],
        }
        val_data = {
            'X': data_dict['X_val_scaled'],
            'y': data_dict['y_val'],
        }
        test_data = {
            'X': data_dict['X_test_scaled'],
            'y': data_dict['y_test'],
        }

        with open(TRAIN_DATA_PATH, 'wb') as f:
            pickle.dump(train_data, f)
        with open(VAL_DATA_PATH, 'wb') as f:
            pickle.dump(val_data, f)
        with open(TEST_DATA_PATH, 'wb') as f:
            pickle.dump(test_data, f)

        # Save scaler
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)

        # Save feature statistics
        with open(FEATURE_STATS_PATH, 'w') as f:
            json.dump(feature_stats, f, indent=2)

        if self.verbose:
            print(f"Train data saved: {TRAIN_DATA_PATH}")
            print(f"Val data saved:   {VAL_DATA_PATH}")
            print(f"Test data saved:  {TEST_DATA_PATH}")
            print(f"Scaler saved:     {SCALER_PATH}")
            print(f"Statistics saved: {FEATURE_STATS_PATH}")

    def load_processed_data(self):
        """Load previously processed data from disk."""
        if not TRAIN_DATA_PATH.exists():
            raise FileNotFoundError(
                "No processed data found. Run preprocessing first.")

        with open(TRAIN_DATA_PATH, 'rb') as f:
            train_data = pickle.load(f)
        with open(VAL_DATA_PATH, 'rb') as f:
            val_data = pickle.load(f)
        with open(TEST_DATA_PATH, 'rb') as f:
            test_data = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(FEATURE_STATS_PATH, 'r') as f:
            feature_stats = json.load(f)

        if self.verbose:
            print("\nLoaded processed data from cache")

        return train_data, val_data, test_data, scaler, feature_stats

    def preprocess(self, features, labels, feature_names=None, force=False):
        """
        Complete preprocessing pipeline.

        Args:
            features: Raw feature matrix
            labels: Label vector
            feature_names: Optional feature names
            force: If True, reprocess even if cached data exists

        Returns:
            tuple: (train_data, val_data, test_data, scaler, feature_stats)
        """
        # Check if processed data exists
        if TRAIN_DATA_PATH.exists() and not force:
            if self.verbose:
                print("\nFound cached processed data. Loading...")
            return self.load_processed_data()

        # Full preprocessing pipeline
        data_dict = self.split_data(features, labels)
        data_dict = self.scale_features(data_dict)
        feature_stats = self.compute_feature_statistics(
            features, labels, feature_names
        )
        self.save_processed_data(data_dict, feature_stats)

        # Prepare output format
        train_data = {
            'X': data_dict['X_train_scaled'], 'y': data_dict['y_train']}
        val_data = {'X': data_dict['X_val_scaled'], 'y': data_dict['y_val']}
        test_data = {'X': data_dict['X_test_scaled'], 'y': data_dict['y_test']}

        if self.verbose:
            print("\n" + "="*70)
            print("Preprocessing Complete")
            print("="*70 + "\n")

        return train_data, val_data, test_data, self.scaler, feature_stats


# --- Compatibility wrappers ---

class TEXBATPreprocessor:
    """
    Wrapper class for compatibility with experiment scripts.
    Maps to GNSSPreprocessor functionality.
    """

    def __init__(self, scaler_type='standard', **kwargs):
        """Initialize preprocessor (scaler_type is kept for compatibility but not used)."""
        self.preprocessor = GNSSPreprocessor(**kwargs)
        self.feature_names = None

    def prepare_features(self, df, engineer=True, clean=True, verbose=True):
        """
        Prepare features from DataFrame with GNSS-specific feature engineering.

        Args:
            df: DataFrame with features and 'label' column
            engineer: Whether to engineer GNSS-specific features
            clean: Whether to clean data (handled by loader)
            verbose: Print information

        Returns:
            DataFrame: Processed DataFrame with engineered features

        References:
            - Kaplan & Hegarty (2017): Understanding GPS/GNSS
            - GNSS signal processing best practices
        """
        # Extract features and labels
        if 'label' not in df.columns and 'Label' not in df.columns:
            raise ValueError("DataFrame must have 'label' column")

        # Standardize label column
        if 'Label' in df.columns:
            df = df.rename(columns={'Label': 'label'})

        df = df.copy()

        if engineer:
            if verbose:
                print("\nEngineering GNSS-specific features...")

            # Get numeric columns (features)
            numeric_cols = df.select_dtypes(
                include=[np.number]).columns.tolist()
            if 'label' in numeric_cols:
                numeric_cols.remove('label')

            # Identify GNSS feature types
            doppler_cols = [
                col for col in numeric_cols if 'doppler' in col.lower()]
            cn0_cols = [col for col in numeric_cols if 'cn0' in col.lower(
            ) or 'c/n0' in col.lower() or 'c_n0' in col.lower()]
            pr_cols = [col for col in numeric_cols if 'pseudorange' in col.lower(
            ) or 'pr' in col.lower()]
            phase_cols = [col for col in numeric_cols if 'phase' in col.lower(
            ) or 'carrier' in col.lower()]

            # 1. Normalized features (if not already normalized)
            for col in doppler_cols:
                if f'{col}_normalized' not in df.columns:
                    col_min, col_max = df[col].min(), df[col].max()
                    if col_max > col_min:
                        df[f'{col}_normalized'] = (
                            df[col] - col_min) / (col_max - col_min)
                        if verbose:
                            print(f"  Created {col}_normalized")

            for col in cn0_cols:
                if f'{col}_normalized' not in df.columns:
                    col_min, col_max = df[col].min(), df[col].max()
                    if col_max > col_min:
                        df[f'{col}_normalized'] = (
                            df[col] - col_min) / (col_max - col_min)
                        if verbose:
                            print(f"  Created {col}_normalized")

            # 2. Interaction features (GNSS signal relationships)
            if doppler_cols and cn0_cols:
                # Use first Doppler column
                for doppler_col in doppler_cols[:1]:
                    for cn0_col in cn0_cols[:1]:  # Use first CN0 column
                        if f'doppler_cn0_ratio' not in df.columns:
                            # Doppler/CN0 ratio (signal quality indicator)
                            df['doppler_cn0_ratio'] = np.abs(
                                df[doppler_col]) / (df[cn0_col] + 1e-10)
                            if verbose:
                                print(f"  Created doppler_cn0_ratio")

            if pr_cols and cn0_cols:
                for pr_col in pr_cols[:1]:
                    for cn0_col in cn0_cols[:1]:
                        if f'pseudorange_cn0_ratio' not in df.columns:
                            # Pseudorange/CN0 ratio
                            df['pseudorange_cn0_ratio'] = df[pr_col] / \
                                (df[cn0_col] + 1e-10)
                            if verbose:
                                print(f"  Created pseudorange_cn0_ratio")

            # 3. Deviation features (Z-scores)
            for col in doppler_cols:
                if f'{col}_deviation' not in df.columns:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    if std_val > 0:
                        df[f'{col}_deviation'] = (df[col] - mean_val) / std_val
                        if verbose:
                            print(f"  Created {col}_deviation")

            for col in cn0_cols:
                if f'{col}_deviation' not in df.columns:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    if std_val > 0:
                        df[f'{col}_deviation'] = (df[col] - mean_val) / std_val
                        if verbose:
                            print(f"  Created {col}_deviation")

            # 4. Binary flags (anomaly indicators)
            for col in doppler_cols:
                if f'is_extreme_{col}' not in df.columns:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    if std_val > 0:
                        df[f'is_extreme_{col}'] = (
                            np.abs(df[col] - mean_val) > 3 * std_val).astype(int)
                        if verbose:
                            print(f"  Created is_extreme_{col}")

            for col in cn0_cols:
                if f'is_low_{col}' not in df.columns:
                    # Low CN0 indicates weak signal (potential spoofing)
                    cn0_threshold = df[col].quantile(0.25)  # Bottom quartile
                    df[f'is_low_{col}'] = (df[col] < cn0_threshold).astype(int)
                    if verbose:
                        print(f"  Created is_low_{col}")

            # 5. Rate of change features (temporal patterns)
            # Note: Requires sorted data by time if available
            time_cols = [
                col for col in numeric_cols if 'time' in col.lower() or 'sec' in col.lower()]
            if time_cols:
                time_col = time_cols[0]
                df_sorted = df.sort_values(
                    time_col) if time_col in df.columns else df

                for col in doppler_cols:
                    if f'{col}_rate' not in df.columns:
                        df_sorted[f'{col}_rate'] = df_sorted[col].diff().fillna(
                            0)
                        if verbose:
                            print(f"  Created {col}_rate")

                for col in cn0_cols:
                    if f'{col}_rate' not in df.columns:
                        df_sorted[f'{col}_rate'] = df_sorted[col].diff().fillna(
                            0)
                        if verbose:
                            print(f"  Created {col}_rate")

                df = df_sorted

            # 6. Signal quality metrics
            if doppler_cols and cn0_cols:
                if 'signal_quality_index' not in df.columns:
                    # Combined signal quality metric
                    doppler_col = doppler_cols[0]
                    cn0_col = cn0_cols[0]
                    # Normalize both to [0,1] and combine
                    doppler_norm = (df[doppler_col] - df[doppler_col].min()) / \
                        (df[doppler_col].max() - df[doppler_col].min() + 1e-10)
                    cn0_norm = (df[cn0_col] - df[cn0_col].min()) / \
                        (df[cn0_col].max() - df[cn0_col].min() + 1e-10)
                    df['signal_quality_index'] = cn0_norm / \
                        (np.abs(doppler_norm) + 1e-10)
                    if verbose:
                        print(f"  Created signal_quality_index")

            if verbose:
                print(
                    f"\nFeature engineering complete. Total features: {len(df.columns) - 1}")

        # Store feature names
        self.feature_names = [col for col in df.columns if col != 'label']

        return df

    def get_feature_names(self, df):
        """Get feature names from DataFrame."""
        if self.feature_names is None:
            self.feature_names = [col for col in df.columns if col != 'label']
        return self.feature_names


def prepare_train_test_split(df, test_size=0.2, random_state=42, verbose=True):
    """
    Prepare train/test split from DataFrame.

    Args:
        df: DataFrame with features and 'label' column
        test_size: Proportion for test set
        random_state: Random seed
        verbose: Print information

    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    # Extract features and labels
    if 'label' not in df.columns and 'Label' not in df.columns:
        raise ValueError("DataFrame must have 'label' column")

    if 'Label' in df.columns:
        df = df.rename(columns={'Label': 'label'})

    # Get only numeric feature columns (exclude label and any non-numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove 'label' from numeric columns if it's numeric
    if 'label' in numeric_cols:
        numeric_cols.remove('label')

    # If label is not numeric, handle it separately
    if 'label' not in numeric_cols:
        # Label might be string like 'normal'/'attack', convert to binary
        y = df['label'].values
        if y.dtype == 'object' or isinstance(y[0], str):
            # Convert string labels to binary (0/1)
            unique_labels = pd.Series(y).unique()
            if len(unique_labels) == 2:
                # Map to 0 and 1
                label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
                y = np.array([label_map[label] for label in y])
                if verbose:
                    print(f"  Converted labels: {label_map}")
            else:
                raise ValueError(
                    f"Expected 2 unique labels, found {len(unique_labels)}: {unique_labels}")
        else:
            y = y.astype(int)
    else:
        y = df['label'].values.astype(int)

    # Use only numeric columns as features
    feature_names = numeric_cols
    X = df[feature_names].values

    # Check for any remaining non-numeric values
    if not np.issubdtype(X.dtype, np.number):
        # Try to convert to numeric, coercing errors to NaN
        X = pd.DataFrame(X, columns=feature_names).apply(
            pd.to_numeric, errors='coerce').values
        # Check for NaN values
        nan_mask = np.isnan(X).any(axis=1)
        if nan_mask.any():
            if verbose:
                print(
                    f"  Warning: {nan_mask.sum()} rows contain non-numeric values, removing them")
            X = X[~nan_mask]
            y = y[~nan_mask]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if verbose:
        print(f"\nTrain set: {len(X_train):,} samples")
        print(f"Test set:  {len(X_test):,} samples")
        print(f"Features:  {len(feature_names)}")
        print(
            f"Label distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    # Test the preprocessor
    print("Testing GNSS Preprocessor...")

    # Create dummy data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 50

    features = np.random.randn(n_samples, n_features)
    labels = np.random.randint(0, 2, n_samples)

    preprocessor = GNSSPreprocessor(verbose=True)
    train, val, test, scaler, stats = preprocessor.preprocess(
        features, labels, force=True
    )

    print("\nPreprocessor test successful!")
    print(f"  Train shape: {train['X'].shape}")
    print(f"  Val shape:   {val['X'].shape}")
    print(f"  Test shape:  {test['X'].shape}")
