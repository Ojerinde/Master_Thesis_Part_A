"""
Attack Configurations
=====================
Parameters for adversarial attacks on GNSS spoofing detection models.

Design Rationale:
- Standard attack methods from adversarial ML literature
- GNSS-specific physical constraints applied
- Attack strength (epsilon) calibrated for signal plausibility
- Epsilon values justified by GNSS signal characteristics

References:
- Goodfellow et al. (2015): Explaining and Harnessing Adversarial Examples
- Madry et al. (2018): Towards Deep Learning Models Resistant to Adversarial Attacks
- Carlini & Wagner (2017): Towards Evaluating the Robustness of Neural Networks
- Kaplan & Hegarty (2017): Understanding GPS/GNSS - Signal characteristics
- Psiaki & Humphreys (2016): GNSS Spoofing and Detection - Attack magnitudes
- Wesson et al. (2011): TEXBAT Dataset - Realistic attack parameters
"""

# --- FGSM (Fast Gradient Sign Method) ---
# Rationale: Single-step gradient-based attack
# Fast, efficient, good baseline for adversarial robustness
# Epsilon values calibrated to normalized feature space (after StandardScaler)
# For GNSS signals: epsilon=0.1 corresponds to ~1 std dev perturbation
# This is physically plausible for measurement noise and small spoofing signals

FGSM_CONFIG = {
    'epsilon': [0.01, 0.05, 0.1, 0.15, 0.2],
    'norm': 'inf',
    'targeted': False,
    'clip_min': None,
    'clip_max': None,
}

# --- PGD (Projected Gradient Descent) ---
# Rationale: Iterative FGSM, stronger attack
# Multi-step optimization, finds better adversarial examples
# Gold standard for adversarial robustness (Madry et al., 2018)
# Step size alpha = epsilon/10 is standard practice

PGD_CONFIG = {
    'epsilon': [0.01, 0.05, 0.1, 0.15, 0.2],
    'alpha': 0.01,
    'num_iter': 40,
    'norm': 'inf',
    'targeted': False,
    'random_start': True,
    'clip_min': None,
    'clip_max': None,
}

# --- C&W Attack (Carlini & Wagner) ---
# Rationale: Optimization-based, minimal perturbation
# Finds smallest perturbation to fool model
# More sophisticated than gradient-based methods

CW_CONFIG = {
    'confidence': 0.0,  # Confidence in misclassification
    'learning_rate': 0.01,
    'binary_search_steps': 9,
    'max_iterations': 1000,
    'abort_early': True,
    'initial_const': 0.01,
    'clip_min': None,
    'clip_max': None,
}

# --- Feature-space attacks (GNSS-specific) ---
# Rationale: Physical plausibility constraints
# Attacks respect GNSS signal physics (e.g., Doppler limits, CN0 ranges)

# Doppler Drift Attack
# Gradual modification of Doppler frequencies (physically plausible)
DOPPLER_DRIFT_CONFIG = {
    'drift_rate': [0.1, 0.5, 1.0, 2.0],  # Hz/s drift rates
    'max_doppler': 5000,  # Hz (typical GNSS Doppler range)
    'min_doppler': -5000,
}

# CN0 Degradation Attack
# Gradual reduction in carrier-to-noise ratio
CN0_DEGRADATION_CONFIG = {
    'degradation_db': [1, 2, 5, 10],  # dB reduction
    'min_cn0': 20,  # Minimum feasible CN0 (dB-Hz)
    'max_cn0': 55,  # Maximum typical CN0
}

# PRN Code Delay Attack
# Slight delays in pseudorandom noise code
PRN_DELAY_CONFIG = {
    'delay_chips': [0.1, 0.5, 1.0, 2.0],  # Chip delays
    'max_delay': 10,  # Maximum reasonable delay
}

# Combined Physical Attack
# Multiple simultaneous GNSS parameter perturbations
COMBINED_PHYSICAL_CONFIG = {
    'doppler_epsilon': 100,  # Hz
    'cn0_epsilon': 2,  # dB
    'delay_epsilon': 0.5,  # chips
}

# --- Attack evaluation settings ---

EVALUATION_CONFIG = {
    'success_threshold': 0.5,  # Confidence threshold for successful attack
    'batch_size': 100,  # Process attacks in batches
    'save_adversarial_examples': True,
    'num_examples_to_save': 100,  # For visualization
}

# --- Transferability analysis ---
# Test if attacks generated on one model fool other models

TRANSFERABILITY_CONFIG = {
    'source_models': ['svm_linear', 'random_forest', 'xgboost'],
    'target_models': ['svm_rbf', 'knn', 'logistic_regression',
                      'cnn_1d', 'lstm', 'transformer'],
    'epsilon': 0.1,  # Fixed epsilon for transferability
    'attack_type': 'pgd',  # Use PGD for stronger attacks
}

# --- Helper functions ---


def get_attack_config(attack_name):
    """Get configuration for a specific attack."""
    config_map = {
        'fgsm': FGSM_CONFIG,
        'pgd': PGD_CONFIG,
        'cw': CW_CONFIG,
        'doppler_drift': DOPPLER_DRIFT_CONFIG,
        'cn0_degradation': CN0_DEGRADATION_CONFIG,
        'prn_delay': PRN_DELAY_CONFIG,
        'combined_physical': COMBINED_PHYSICAL_CONFIG,
    }

    if attack_name.lower() not in config_map:
        raise ValueError(f"Unknown attack: {attack_name}")

    return config_map[attack_name.lower()].copy()


def set_feature_bounds(clip_min, clip_max):
    """
    Set feature bounds for all attacks based on data statistics.

    Args:
        clip_min: Minimum feature values (array)
        clip_max: Maximum feature values (array)
    """
    for config in [FGSM_CONFIG, PGD_CONFIG, CW_CONFIG]:
        config['clip_min'] = clip_min
        config['clip_max'] = clip_max


if __name__ == "__main__":
    print("Available attack configurations:")
    attacks = ['fgsm', 'pgd', 'cw', 'doppler_drift',
               'cn0_degradation', 'prn_delay', 'combined_physical']

    for attack in attacks:
        config = get_attack_config(attack)
        print(f"\n{attack.upper()}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
