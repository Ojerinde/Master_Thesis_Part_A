"""
GNSS Feature-Space Attacks
============================
Model-agnostic attacks that operate directly in feature space without
requiring model gradients.

Attack taxonomy
---------------
DLSA — Data Location Shift Attack
    Coherent adversarial bias toward the spoofing class centroid.
    Direction is computed from class centroids — always adversarial.

SNA — Similarity-based Noise Attack
    Structured noise projected onto an L2 epsilon ball. Seeded RNG.

TPA — Temporal Pattern Attack
    Smooth temporal modulation of Doppler (carry-off) and CN0 (flicker).

References
----------
- Kaplan & Hegarty (2017): Understanding GPS/GNSS, 3rd ed.
- Psiaki & Humphreys (2016): GNSS Spoofing and Detection. Proc. IEEE.
"""

import numpy as np
from typing import Optional, List

from utils.gnss_constraints import GNSSConstraintEnforcer


# --- DLSA — Data Location Shift Attack ---

class DataLocationShiftAttack:
    """
    Coherent adversarial bias toward the spoofing class centroid.

    Parameters
    ----------
    shift_scale   : perturbation magnitude (normalised std devs)
    feature_names : used to select location-sensitive feature columns
    gnss_enforcer : pre-fitted GNSSConstraintEnforcer
    seed          : kept for API consistency (DLSA is deterministic given y)
    """

    def __init__(self, shift_scale: float = 0.10,
                 feature_names: Optional[List[str]] = None,
                 gnss_enforcer: Optional[GNSSConstraintEnforcer] = None,
                 seed: int = 42):
        self.shift_scale = shift_scale
        self.feature_names = feature_names or []
        self.enforcer = gnss_enforcer
        self.seed = seed

        self._loc_idx = [
            i for i, n in enumerate(self.feature_names)
            if any(k in n.lower() for k in
                   ('pseudorange', 'elevation', 'azimuth'))
        ]

    def generate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Apply adversarial coherent location shift.

        Direction = sign(centroid_spoof - centroid_normal) per feature.
        This always pushes normal samples toward spoofing feature space.
        Falls back to all-ones if y contains only one class.
        """
        X_adv = X.copy().astype(np.float32)

        has_both = (y is not None and
                    (y == 0).any() and (y == 1).any())

        if has_both:
            centroid_spoof = X[y == 1].mean(axis=0)
            centroid_normal = X[y == 0].mean(axis=0)
            direction = np.sign(centroid_spoof - centroid_normal
                                ).astype(np.float32)
            direction[direction == 0] = 1.0  # tie-break toward spoofing
        else:
            direction = np.ones(X.shape[1], dtype=np.float32)

        bias = direction * self.shift_scale

        if self._loc_idx:
            for idx in self._loc_idx:
                if idx < X.shape[1]:
                    X_adv[:, idx] += bias[idx]
        else:
            # No named location features: mild global shift
            X_adv += bias * 0.5

        if self.enforcer is not None:
            X_adv = self.enforcer.clip_to_gnss_bounds(X_adv)

        return X_adv


# --- SNA — Similarity-based Noise Attack ---

class SimilarityNoiseAttack:
    """
    Structured noise projected onto an L2 epsilon ball.

    Parameters
    ----------
    epsilon       : L2 norm budget (normalised space)
    gnss_enforcer : pre-fitted GNSSConstraintEnforcer
    seed          : random seed — ensures same perturbation every call
    """

    def __init__(self, epsilon: float = 0.10,
                 gnss_enforcer: Optional[GNSSConstraintEnforcer] = None,
                 seed: int = 42):
        self.epsilon = epsilon
        self.enforcer = gnss_enforcer
        self.seed = seed
        self._std = None

    def fit(self, X_train: np.ndarray):
        """Compute per-feature std from clean training data."""
        self._std = np.std(X_train, axis=0) + 1e-8
        return self

    def generate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate manifold-aligned adversarial perturbations."""
        rng = np.random.default_rng(self.seed)

        if self._std is not None:
            z = rng.standard_normal(X.shape).astype(np.float32)
            noise = z * self._std
        else:
            noise = rng.standard_normal(X.shape).astype(np.float32)

        norms = np.linalg.norm(noise, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        noise = noise / norms * self.epsilon

        X_adv = (X + noise).astype(np.float32)

        if self.enforcer is not None:
            X_adv = self.enforcer.clip_to_gnss_bounds(X_adv)

        return X_adv


# --- TPA — Temporal Pattern Attack ---

class TemporalPatternAttack:
    """
    Smooth temporal modulation of Doppler and CN0 features.

    Physical interpretation
    -----------------------
    Doppler ramp  : mimics carry-off spoofing.
    CN0 flicker   : mimics power-controlled spoofing / jamming.

    Parameters
    ----------
    doppler_amp   : ramp amplitude (normalised std devs)
    cn0_amp       : flicker amplitude (normalised std devs)
    feature_names : identifies Doppler and CN0 columns
    gnss_enforcer : pre-fitted GNSSConstraintEnforcer
    seed          : random seed for Doppler noise component
    """

    def __init__(self, doppler_amp: float = 0.10,
                 cn0_amp: float = 0.05,
                 feature_names: Optional[List[str]] = None,
                 gnss_enforcer: Optional[GNSSConstraintEnforcer] = None,
                 seed: int = 42):
        self.doppler_amp = doppler_amp
        self.cn0_amp = cn0_amp
        self.feature_names = feature_names or []
        self.enforcer = gnss_enforcer
        self.seed = seed

        self._doppler_idx = [
            i for i, n in enumerate(self.feature_names)
            if 'doppler' in n.lower()
        ]
        self._cn0_idx = [
            i for i, n in enumerate(self.feature_names)
            if 'cn0' in n.lower() or 'c_n0' in n.lower()
        ]

    def generate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Apply Doppler ramp + CN0 flicker. Rows assumed time-ordered."""
        n = X.shape[0]
        X_adv = X.copy().astype(np.float32)
        rng = np.random.default_rng(self.seed)

        if self._doppler_idx:
            ramp = np.linspace(-self.doppler_amp, self.doppler_amp,
                               n, dtype=np.float32)
            noise = (rng.standard_normal(n) * 0.05 * self.doppler_amp
                     ).astype(np.float32)
            for idx in self._doppler_idx:
                if idx < X.shape[1]:
                    X_adv[:, idx] += ramp + noise

        if self._cn0_idx:
            t = np.linspace(0, 2 * np.pi, n, dtype=np.float32)
            flicker = (self.cn0_amp * np.sin(t)).astype(np.float32)
            for idx in self._cn0_idx:
                if idx < X.shape[1]:
                    X_adv[:, idx] += flicker

        if self.enforcer is not None:
            X_adv = self.enforcer.clip_to_gnss_bounds(X_adv)

        return X_adv
