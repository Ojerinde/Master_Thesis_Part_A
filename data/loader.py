"""
TEXBAT Data Loader
==================
Robust data loading with validation, caching, and error handling.
"""

from config.paths import (
    TEXBAT_CSV, TEXBAT_TRACK_CSV, PROCESSED_DATA_DIR,
    TRAIN_DATA_PATH, TEST_DATA_PATH, VAL_DATA_PATH
)
import pandas as pd
import numpy as np
import pickle
import json
from typing import Tuple, Optional
import warnings

# Cosmetic: models are fit on named DataFrames but predicted on NumPy arrays
# (columns are in the same fixed order, so results are identical). Silence the
# sklearn "X does not have valid feature names" nag so run logs stay readable.
# Imported by every experiment, so this registers process-wide before predict().
warnings.filterwarnings(
    "ignore", message=".*does not have valid feature names.*")


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


# ============================================================================
# FGI-GSRx TRACK-ONLY CORPUS (Paper 1 rebuild)
# ============================================================================
# Physical receiver observables ONLY. Metadata/time/identity columns
# (scenario, source_file, spoof_type, prn, epoch_idx, t_sec, segment, label,
# label_name) are deliberately EXCLUDED from the feature set: t_sec/epoch_idx
# encode the spoof onset and prn/scenario encode recording identity, so using
# them would reproduce the WP2 timestamp/recording leakage. Also excluded are
# fll_filter (FLL loop-filter output; NaN in steady-state fine tracking) and the
# ALGEBRAICALLY DERIVED observables, which would make feature-space adversarial
# attacks physically UNREALIZABLE if perturbed independently of their parents
# (verified on the corpus, 2026-07-09):
#     carr_freq_hz     == doppler_hz              (identical; std of diff = 0.0)
#     prompt_power     == i_prompt^2 + q_prompt^2 (exact)
#     prompt_phase_rad == atan2(q_prompt, i_prompt) (exact)
# The instantaneous correlator state is kept as (i_prompt, q_prompt); the residual
# SOFT coupling C/N0 ~ f(I/Q power) is enforced physically by the
# GNSSConstraintEnforcer during attacks. Only these 9 physically INDEPENDENT
# signal-domain observables may be fed to a detector. See memory iq-feature-coupling.
TEXBAT_TRACK_FEATURES = [
    'cn0_dbhz', 'mean_cn0_dbhz', 'noise_cn0', 'doppler_hz',
    'i_prompt', 'q_prompt',
    'dll_discr', 'pll_lock', 'fll_lock',
]


def load_texbat_track(verbose=True, validate=True, scenarios=None):
    """
    Load the FGI-GSRx track-only observable corpus (Paper 1 rebuild).

    Produced by fgi/export_texbat_track.m from raw TEXBAT tracked through the
    FGI-GSRx software receiver. Labels: 0=genuine, 1=counterfeit, from the
    within-recording spoofing onset (no recording-identity confound).

    Args:
        verbose   : print progress
        validate  : drop rows with NaN/Inf features or out-of-range C/N0
        scenarios : optional list to keep (e.g. ['cleanstatic','ds2','ds3','ds7'])

    Returns:
        df            : DataFrame (feature columns + metadata columns retained)
        feature_names : the 9 physically-independent observable columns (TEXBAT_TRACK_FEATURES)
    """
    if not TEXBAT_TRACK_CSV.exists():
        raise FileNotFoundError(
            f"TEXBAT track corpus not found: {TEXBAT_TRACK_CSV}\n"
            f"   Generate it with fgi/export_texbat_track.m after tracking the "
            f"TEXBAT scenarios in FGI-GSRx."
        )

    df = pd.read_csv(TEXBAT_TRACK_CSV)
    if verbose:
        print(f"Loaded {len(df):,} rows x {df.shape[1]} cols from {TEXBAT_TRACK_CSV.name}")

    if scenarios is not None:
        df = df[df['scenario'].isin(scenarios)].reset_index(drop=True)
        if verbose:
            print(f"  filtered to scenarios {scenarios}: {len(df):,} rows")

    if 'label' not in df.columns:
        raise ValueError("No 'label' column in TEXBAT track corpus")
    missing = [c for c in TEXBAT_TRACK_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Corpus missing expected feature columns: {missing}")

    if validate:
        n0 = len(df)
        feat = df[TEXBAT_TRACK_FEATURES]
        bad = feat.isnull().any(axis=1)
        bad = bad | np.isinf(feat.to_numpy(dtype=float)).any(axis=1)
        bad = bad | (df['cn0_dbhz'] <= 0) | (df['cn0_dbhz'] > 70)   # dB-Hz physical range
        if bad.sum() > 0:
            df = df[~bad].reset_index(drop=True)
        if verbose:
            print(f"  validation: {len(df):,} rows retained "
                  f"({n0 - len(df):,} removed for NaN/Inf/C-N0 out of range)")

    if verbose:
        vc = df['label'].value_counts().to_dict()
        print(f"  labels: genuine(0)={vc.get(0, 0):,}  spoof(1)={vc.get(1, 0):,}")
        print(f"  scenarios: {sorted(df['scenario'].unique().tolist())}")
        print(f"  features ({len(TEXBAT_TRACK_FEATURES)}): {TEXBAT_TRACK_FEATURES}")

    return df, TEXBAT_TRACK_FEATURES


def load_track_splits(scenarios=None, train_frac=0.70, val_frac=0.10,
                      purge=20, verbose=False):
    """Leakage-free, block-temporal train/val/test split of the FGI corpus,
    shared by every experiment script so the partition is identical everywhere.

    A RANDOM split of per-epoch tracking data LEAKS: adjacent epochs of the same
    PRN are near-identical (loop time constants ~0.1-1 s), so shuffling scatters
    almost-duplicate rows across train and test and inflates every metric. The
    standard remedy for autocorrelated series is a block-temporal split with
    purging (cf. time-series CV; Lopez de Prado purging). Here each contiguous run
    -- one (scenario, prn, segment) group ordered by t_sec -- is cut into
    CONTIGUOUS blocks: first `train_frac` -> train, next `val_frac` -> val,
    remainder -> test, DROPPING `purge` epochs (~1 s at the 20 Hz export cadence)
    at each block boundary to break short-range autocorrelation. Every scenario /
    PRN / class appears in all three splits, so the partition is representative
    without leaking. Fully deterministic (no RNG).

    (For the cross-scenario generalization test, call with `scenarios=` to train
    on one set and evaluate on a held-out scenario.)

    Returns:
        X_train, X_val, X_test : UNSCALED features (the 9 independent observables).
                                 Classical Pipelines scale internally; for the DL
                                 models apply `scaler.transform(...)`.
        y_train, y_val, y_test : int labels (0=genuine, 1=spoof).
        feature_names          : the observable column names.
        scaler                 : StandardScaler fitted on the train block.
    """
    from sklearn.preprocessing import StandardScaler

    df, feats = load_texbat_track(verbose=verbose, validate=True,
                                  scenarios=scenarios)
    df = df.reset_index(drop=True)
    tr, va, te = [], [], []
    for _, g in df.groupby(['scenario', 'prn', 'segment'], sort=False):
        idx = g.sort_values('t_sec').index.to_numpy()
        n = len(idx)
        n_tr = int(round(n * train_frac))
        n_va = int(round(n * val_frac))
        tr.append(idx[:max(0, n_tr - purge)])
        va.append(idx[n_tr + purge:max(n_tr + purge, n_tr + n_va - purge)])
        te.append(idx[n_tr + n_va + purge:])
    tr = np.concatenate(tr)
    va = np.concatenate(va)
    te = np.concatenate(te)
    X = df[feats].values.astype(np.float64)
    y = df['label'].values.astype(int)
    scaler = StandardScaler().fit(X[tr])
    if verbose:
        print(f"  block-temporal split (purge={purge} epochs): "
              f"train={len(tr):,}  val={len(va):,}  test={len(te):,}")
    return (X[tr], X[va], X[te], y[tr], y[va], y[te], feats, scaler)


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
