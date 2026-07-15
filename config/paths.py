"""
Centralised path configuration for the GNSS adversarial research project.
"""

from pathlib import Path

# Project root (derived from this file's location)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw data
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
TEXBAT_CSV = RAW_DATA_DIR / "texbat_channel_combined.csv"

# Processed / cached splits
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
TEXBAT_TRACK_CSV = PROCESSED_DATA_DIR / "texbat_track_combined.csv"  # FGI-GSRx observable corpus (Paper 1 rebuild)
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_data.pkl"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test_data.pkl"
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val_data.pkl"
FEATURE_STATS_PATH = PROCESSED_DATA_DIR / "feature_statistics.json"
SCALER_PATH = PROCESSED_DATA_DIR / "scaler.pkl"

# Results root
OUTPUT_ROOT = PROJECT_ROOT / "results"

# Saved models
MODELS_DIR = OUTPUT_ROOT / "models"
CLASSICAL_MODELS = MODELS_DIR / "classical"
DL_MODELS = MODELS_DIR / "deep_learning"

# Figures (paper-ready)
FIGURES_DIR = OUTPUT_ROOT / "figures"
BASELINE_FIGURES = FIGURES_DIR / "baseline"
ADVERSARIAL_FIGURES = FIGURES_DIR / "adversarial"
ANALYSIS_FIGURES = FIGURES_DIR / "analysis"

# Tables (CSV for LaTeX import)
TABLES_DIR = OUTPUT_ROOT / "tables"
BASELINE_RESULTS_PATH = TABLES_DIR / "baseline_results.csv"
ADVERSARIAL_RESULTS_PATH = TABLES_DIR / "adversarial_results.csv"
PERFORMANCE_MATRIX_PATH = TABLES_DIR / "performance_matrix.csv"

# Logs
LOGS_DIR = OUTPUT_ROOT / "logs"
TRAINING_LOG = LOGS_DIR / "training.log"
EVALUATION_LOG = LOGS_DIR / "evaluation.log"
ATTACK_LOG = LOGS_DIR / "attack.log"

# Experiment checkpoints
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"


def create_directories():
    """Create all output directories if they do not exist."""
    for d in [
        PROCESSED_DATA_DIR, RAW_DATA_DIR,
        CLASSICAL_MODELS, DL_MODELS,
        BASELINE_FIGURES, ADVERSARIAL_FIGURES, ANALYSIS_FIGURES,
        TABLES_DIR, LOGS_DIR, CHECKPOINT_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
    print("All directories created successfully")


def verify_data_paths():
    """Raise if the raw TEXBAT CSV is missing."""
    if not TEXBAT_CSV.exists():
        raise FileNotFoundError(
            f"TEXBAT data not found at: {TEXBAT_CSV}\n"
            f"Place texbat_channel_combined.csv in {RAW_DATA_DIR}"
        )
    print(f"Data found: {TEXBAT_CSV}")
    return True


if __name__ == "__main__":
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data file    : {TEXBAT_CSV}")
    print(f"Output dir   : {OUTPUT_ROOT}")
    create_directories()
    verify_data_paths()
    print("Configuration verified.")
