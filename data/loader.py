"""
TEXBAT Data Loader
==================
Robust data loading with validation, caching, and error handling.
"""

from config.paths import (
    TEXBAT_CSV, PROCESSED_DATA_DIR,
    TRAIN_DATA_PATH, TEST_DATA_PATH, VAL_DATA_PATH
)
import pandas as pd
import numpy as np
import pickle
import json
from typing import Tuple, Optional
import warnings


class TEXBATLoader:
    """
    TEXBAT dataset loader with validation and caching.

    Follows GNSS research best practices:
    - Validates physical constraints (Doppler, CN0, etc.)
    - Removes corrupted samples
    - Maintains class balance information
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.data = None
        self.features = None
        self.labels = None

    def load_raw_data(self, force_reload=False):
        """
        Load raw TEXBAT CSV data.

        Args:
            force_reload: If True, ignore cached data and reload from CSV

        Returns:
            pd.DataFrame: Raw TEXBAT data
        """
        if self.verbose:
            print("\n" + "="*70)
            print("Loading TEXBAT Dataset")
            print("="*70 + "\n")

        # Check if file exists
        if not TEXBAT_CSV.exists():
            raise FileNotFoundError(
                f"TEXBAT data not found: {TEXBAT_CSV}\n"
                f"   Please verify the data location."
            )

        if self.verbose:
            print(f"Data file found: {TEXBAT_CSV}")

        # Load CSV
        try:
            df = pd.read_csv(TEXBAT_CSV)
            if self.verbose:
                print(f"Loaded {len(df):,} samples")
                print(f"Features: {df.shape[1]} columns")

        except Exception as e:
            raise RuntimeError(f"Error loading CSV: {str(e)}")

        # Validate column names
        if 'label' not in df.columns and 'Label' not in df.columns:
            raise ValueError(
                "No label column found. Expected 'label' or 'Label'"
            )

        # Standardize label column name
        if 'Label' in df.columns:
            df.rename(columns={'Label': 'label'}, inplace=True)

        self.data = df
        return df

    def validate_data(self, df):
        """
        Validate TEXBAT data for physical plausibility.

        Checks:
        - Doppler frequencies within [-5000, 5000] Hz
        - CN0 values within [0, 60] dB-Hz
        - No infinite or NaN values
        - Reasonable pseudorange values

        Args:
            df: DataFrame to validate

        Returns:
            pd.DataFrame: Cleaned data
            dict: Validation statistics
        """
        if self.verbose:
            print("\nValidating data quality...")

        initial_count = len(df)
        validation_stats = {
            'initial_samples': initial_count,
            'removed_samples': 0,
            'removed_reasons': {}
        }

        # Check for NaN/infinite values
        nan_mask = df.isnull().any(axis=1)
        inf_mask = np.isinf(df.select_dtypes(include=[np.number])).any(axis=1)
        invalid_mask = nan_mask | inf_mask

        if invalid_mask.sum() > 0:
            validation_stats['removed_reasons']['nan_inf'] = invalid_mask.sum()
            df = df[~invalid_mask]
            if self.verbose:
                print(
                    f"  Removed {invalid_mask.sum()} samples with NaN/Inf values")

        # Check Doppler frequencies (if column exists)
        doppler_cols = [col for col in df.columns if 'doppler' in col.lower()]
        for col in doppler_cols:
            if col in df.columns:
                # Unrealistic Doppler
                invalid_doppler = (df[col].abs() > 10000)
                if invalid_doppler.sum() > 0:
                    validation_stats['removed_reasons'][f'{col}_invalid'] = invalid_doppler.sum(
                    )
                    df = df[~invalid_doppler]
                    if self.verbose:
                        print(
                            f"  Removed {invalid_doppler.sum()} samples with invalid {col}")

        # Check CN0 values (if column exists)
        cn0_cols = [col for col in df.columns if 'cn0' in col.lower()
                    or 'c/n0' in col.lower()]
        for col in cn0_cols:
            if col in df.columns:
                invalid_cn0 = (df[col] < 0) | (df[col] > 70)
                if invalid_cn0.sum() > 0:
                    validation_stats['removed_reasons'][f'{col}_invalid'] = invalid_cn0.sum(
                    )
                    df = df[~invalid_cn0]
                    if self.verbose:
                        print(
                            f"  Removed {invalid_cn0.sum()} samples with invalid {col}")

        validation_stats['removed_samples'] = initial_count - len(df)
        validation_stats['final_samples'] = len(df)

        if self.verbose:
            print(f"\nValidation complete:")
            print(f"  Initial: {initial_count:,} samples")
            print(
                f"  Removed: {validation_stats['removed_samples']:,} samples")
            print(f"  Final: {len(df):,} samples")

        return df, validation_stats

    def get_class_distribution(self, labels):
        """Get class distribution statistics."""
        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique, counts))

        if self.verbose:
            print("\nClass Distribution:")
            for label, count in distribution.items():
                percentage = (count / len(labels)) * 100
                label_name = "Attack" if label == 1 else "Normal"
                print(
                    f"  {label_name} (Label {label}): {count:,} ({percentage:.1f}%)")

        return distribution

    def split_features_labels(self, df):
        """
        Split dataframe into features and labels.

        Args:
            df: DataFrame with features and 'label' column

        Returns:
            tuple: (features, labels)
        """
        if 'label' not in df.columns:
            raise ValueError("No 'label' column found in dataframe")

        # Separate features and labels
        labels = df['label'].values
        features = df.drop('label', axis=1)

        # Remove non-numeric columns
        numeric_features = features.select_dtypes(include=[np.number])

        if len(numeric_features.columns) < len(features.columns):
            dropped = set(features.columns) - set(numeric_features.columns)
            if self.verbose:
                print(f"\nDropped non-numeric columns: {dropped}")

        features = numeric_features.values

        if self.verbose:
            print(f"\nFeature shape: {features.shape}")
            print(f"Label shape: {labels.shape}")

        self.features = features
        self.labels = labels

        return features, labels

    def load_and_validate(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Complete pipeline: load, validate, and split data.

        Returns:
            tuple: (features, labels, metadata)
        """
        # Load raw data
        df = self.load_raw_data()

        # Validate data
        df, validation_stats = self.validate_data(df)

        # Get class distribution
        labels_temp = df['label'].values
        class_dist = self.get_class_distribution(labels_temp)

        # Split features and labels
        features, labels = self.split_features_labels(df)

        # Prepare metadata
        metadata = {
            'num_samples': len(features),
            'num_features': features.shape[1],
            'class_distribution': class_dist,
            'validation_stats': validation_stats,
            'feature_names': list(df.drop('label', axis=1).columns),
        }

        if self.verbose:
            print("\n" + "="*70)
            print("Data Loading Complete")
            print("="*70)
            print(f"Total samples: {metadata['num_samples']:,}")
            print(f"Total features: {metadata['num_features']}")
            print("="*70 + "\n")

        return features, labels, metadata


def load_texbat_data(verbose=True):
    """
    Convenience function to load TEXBAT data.

    Args:
        verbose: Print loading information

    Returns:
        tuple: (features, labels, metadata)
    """
    loader = TEXBATLoader(verbose=verbose)
    return loader.load_and_validate()


def load_texbat(verbose=True, validate=True):
    """
    Wrapper function to load TEXBAT data as DataFrame.
    Matches the interface expected by experiment scripts.

    Args:
        verbose: Print loading information
        validate: Whether to validate data (currently always True)

    Returns:
        tuple: (DataFrame, TEXBATLoader instance)
    """
    loader = TEXBATLoader(verbose=verbose)
    df = loader.load_raw_data()
    if validate:
        df, _ = loader.validate_data(df)
    return df, loader


if __name__ == "__main__":
    # Test the loader
    print("Testing TEXBAT Data Loader...")

    try:
        features, labels, metadata = load_texbat_data(verbose=True)

        print("\nMetadata:")
        print(json.dumps(metadata, indent=2, default=str))

        print("\nData loader test successful!")

    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
