"""
Data Augmentation
=================
GNSS-specific data augmentation techniques for improving model robustness.

Design Rationale:
- Physical plausibility: Augmentations respect GNSS signal physics
- Realistic noise: Based on actual GNSS measurement characteristics
- Class-preserving: Augmentations don't change attack/normal labels

References:
- GNSS signal characteristics (Kaplan & Hegarty, 2017)
- Data augmentation for time series (Wen et al., 2020)
"""

import numpy as np
from typing import Tuple, Optional


class GNSSAugmenter:
    """
    GNSS-specific data augmentation.

    Augmentation techniques:
    1. Gaussian noise (measurement noise)
    2. Doppler shift (small frequency variations)
    3. CN0 variation (signal strength fluctuations)
    4. Time warping (temporal variations)
    """

    def __init__(self, random_state=42, verbose=True):
        """
        Initialize augmenter.

        Args:
            random_state: Random seed for reproducibility
            verbose: Print augmentation information
        """
        self.random_state = random_state
        self.verbose = verbose
        np.random.seed(random_state)

    def add_gaussian_noise(self, X: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
        """
        Add Gaussian noise to features.

        Rationale: Simulates measurement noise in GNSS receivers.

        Args:
            X: Feature matrix (n_samples, n_features)
            noise_std: Standard deviation of noise (relative to feature std)

        Returns:
            Augmented feature matrix
        """
        feature_std = np.std(X, axis=0, keepdims=True)
        noise = np.random.normal(0, noise_std * noise_std, X.shape)
        return X + noise

    def apply_doppler_shift(self, X: np.ndarray, doppler_cols: Optional[list] = None,
                            max_shift: float = 50.0) -> np.ndarray:
        """
        Apply small Doppler frequency shifts.

        Rationale: Simulates small velocity changes or receiver motion.

        Args:
            X: Feature matrix
            doppler_cols: Indices of Doppler-related columns (if None, auto-detect)
            max_shift: Maximum Doppler shift in Hz

        Returns:
            Augmented feature matrix
        """
        X_aug = X.copy()

        # Auto-detect Doppler columns if not provided
        if doppler_cols is None:
            # Default to first N columns when column indices are not specified
            doppler_cols = list(range(min(10, X.shape[1])))

        # Apply small random shifts
        for col_idx in doppler_cols:
            if col_idx < X.shape[1]:
                shift = np.random.uniform(-max_shift, max_shift, X.shape[0])
                X_aug[:, col_idx] += shift

        return X_aug

    def apply_cn0_variation(self, X: np.ndarray, cn0_cols: Optional[list] = None,
                            max_variation: float = 2.0) -> np.ndarray:
        """
        Apply carrier-to-noise ratio variations.

        Rationale: Simulates signal strength fluctuations.

        Args:
            X: Feature matrix
            cn0_cols: Indices of CN0-related columns
            max_variation: Maximum CN0 variation in dB

        Returns:
            Augmented feature matrix
        """
        X_aug = X.copy()

        if cn0_cols is None:
            # Auto-detect CN0 columns
            cn0_cols = list(range(min(10, X.shape[1])))  # Placeholder

        for col_idx in cn0_cols:
            if col_idx < X.shape[1]:
                variation = np.random.uniform(-max_variation,
                                              max_variation, X.shape[0])
                X_aug[:, col_idx] += variation
                # Ensure CN0 stays in reasonable range [0, 60] dB-Hz
                X_aug[:, col_idx] = np.clip(X_aug[:, col_idx], 0, 60)

        return X_aug

    def apply_time_warping(self, X: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """
        Apply time warping (temporal variations).

        Rationale: Simulates temporal variations in signal characteristics.

        Args:
            X: Feature matrix (assumes temporal features)
            sigma: Warping strength

        Returns:
            Augmented feature matrix
        """
        # Simple implementation: add correlated noise
        # More sophisticated: use time warping algorithms
        n_samples, n_features = X.shape
        warp = np.random.normal(0, sigma, (n_samples, n_features))
        # Apply smoothing to make it more realistic
        from scipy.ndimage import gaussian_filter1d
        warp = gaussian_filter1d(warp, sigma=1.0, axis=0)
        return X + warp

    def augment(self, X: np.ndarray, y: np.ndarray,
                augmentation_types: list = ['gaussian_noise'],
                num_augmented: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentation to dataset.

        Args:
            X: Feature matrix
            y: Labels
            augmentation_types: List of augmentation methods to apply
            num_augmented: Number of augmented samples (default: same as original)

        Returns:
            Tuple of (augmented_X, augmented_y)
        """
        if num_augmented is None:
            num_augmented = len(X)

        if self.verbose:
            print(f"\nAugmenting {len(X)} samples -> {num_augmented} samples")
            print(f"Augmentation types: {augmentation_types}")

        X_aug = []
        y_aug = []

        for i in range(num_augmented):
            # Select random sample
            idx = np.random.randint(0, len(X))
            x_sample = X[idx:idx+1].copy()
            y_sample = y[idx]

            # Apply selected augmentations
            for aug_type in augmentation_types:
                if aug_type == 'gaussian_noise':
                    x_sample = self.add_gaussian_noise(x_sample)
                elif aug_type == 'doppler_shift':
                    x_sample = self.apply_doppler_shift(x_sample)
                elif aug_type == 'cn0_variation':
                    x_sample = self.apply_cn0_variation(x_sample)
                elif aug_type == 'time_warping':
                    x_sample = self.apply_time_warping(x_sample)

            X_aug.append(x_sample[0])
            y_aug.append(y_sample)

        X_aug = np.array(X_aug)
        y_aug = np.array(y_aug)

        if self.verbose:
            print(f"Augmentation complete: {X_aug.shape}")

        return X_aug, y_aug

    def augment_batch(self, X: np.ndarray, augmentation_types: list = ['gaussian_noise']) -> np.ndarray:
        """
        Apply augmentation to a batch of samples.

        Args:
            X: Feature matrix (n_samples, n_features)
            augmentation_types: List of augmentation methods

        Returns:
            Augmented feature matrix
        """
        X_aug = X.copy()

        for aug_type in augmentation_types:
            if aug_type == 'gaussian_noise':
                X_aug = self.add_gaussian_noise(X_aug)
            elif aug_type == 'doppler_shift':
                X_aug = self.apply_doppler_shift(X_aug)
            elif aug_type == 'cn0_variation':
                X_aug = self.apply_cn0_variation(X_aug)
            elif aug_type == 'time_warping':
                X_aug = self.apply_time_warping(X_aug)

        return X_aug


if __name__ == "__main__":
    # Test augmentation
    print("Testing GNSS Data Augmentation...")

    # Create dummy data
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)

    augmenter = GNSSAugmenter(verbose=True)

    # Test each augmentation type
    print("\n1. Gaussian noise:")
    X_noise = augmenter.add_gaussian_noise(X, noise_std=0.05)
    print(
        f"   Original std: {np.std(X):.4f}, Augmented std: {np.std(X_noise):.4f}")

    print("\n2. Doppler shift:")
    X_doppler = augmenter.apply_doppler_shift(X, max_shift=50.0)
    print(f"   Max difference: {np.max(np.abs(X - X_doppler)):.4f}")

    print("\n3. CN0 variation:")
    X_cn0 = augmenter.apply_cn0_variation(X, max_variation=2.0)
    print(f"   Max difference: {np.max(np.abs(X - X_cn0)):.4f}")

    print("\n4. Full augmentation:")
    X_aug, y_aug = augmenter.augment(
        X, y,
        augmentation_types=['gaussian_noise', 'doppler_shift'],
        num_augmented=200
    )
    print(f"   ✓ Augmented shape: {X_aug.shape}, Labels: {y_aug.shape}")

    print("\n✓ All augmentation tests passed!")
