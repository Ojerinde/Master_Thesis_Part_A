# GNSS Adversarial Machine Learning Research

Adversarial robustness analysis of classical and deep learning GNSS spoofing detectors on the TEXBAT dataset.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (classical → DL → adversarial → statistics)
python main.py

# Run specific stages
python main.py --stage 1        # Classical baselines only
python main.py --stage 1 2      # Classical + DL baselines
python main.py --stage 3 --quick  # Adversarial eval (2000 samples, ~10 min)
```

## Pipeline

| Stage | Script                                     | Description                                                                                    |
| ----- | ------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| 1     | `experiments/01_classical_baseline.py`     | Train 17 classical models (7 base × default/balanced/SMOTE), 5-fold CV, threshold optimization |
| 2     | `experiments/02_deep_learning_baseline.py` | Train 6 DL architectures (CNN-1D, LSTM, BiLSTM, CNN-LSTM, Transformer, TCN) with 3 seeds       |
| 3     | `experiments/03_adversarial_evaluation.py` | Run 7 attack types (FGSM, PGD, transfer, DLSA, SNA, TPA) at 3 epsilon levels                   |
| 4     | `experiments/04_statistical_analysis.py`   | Bootstrap CIs, McNemar, Wilcoxon, Friedman tests, Cohen's d                                    |

Stages must run in order (each depends on artifacts from the previous).

## Dataset

**TEXBAT** (Texas Spoofing Test Battery): ~94,900 samples, 23 engineered features.

- Place `texbat_channel_combined.csv` in `data/raw/`
- Class distribution: 33.6% normal / 66.4% attack

## Project Structure

```
gnss_adversarial_research/
├── main.py                  # Pipeline entry point
├── config/                  # Paths, model configs, attack configs
├── data/
│   ├── raw/                 # TEXBAT CSV
│   ├── processed/           # Saved scaler, feature engineer
│   ├── loader.py            # TEXBATLoader
│   ├── preprocessor.py      # Splitting, scaling, caching
│   └── feature_engineering.py  # SafeFeatureEngineer (no leakage)
├── models/
│   ├── classical/           # SVM, RF, XGBoost, LightGBM, etc.
│   └── deep_learning/       # CNN-1D, LSTM, BiLSTM, CNN-LSTM, Transformer, TCN
├── attacks/
│   ├── fgsm.py, pgd.py, cw.py  # Gradient-based attacks
│   └── gnss_attacks.py      # DLSA, SNA, TPA (GNSS-specific)
├── evaluation/              # Metrics, visualization, statistical tests
├── experiments/             # Stage scripts (01–04)
├── results/
│   ├── models/              # Saved trained models (.joblib, .pt)
│   ├── tables/              # CSV results
│   ├── figures/             # Generated plots
│   └── checkpoints/         # Training checkpoints
└── utils/                   # Logging, checkpoints, GNSS constraint enforcer
```

## Key Design Decisions

- **No data leakage**: `SafeFeatureEngineer` fits only on training set; scaler saved and reused across stages.
- **Dual data paths in Stage 3**: Classical models (sklearn Pipelines) receive unscaled input (they scale internally). DL models receive externally scaled input. This prevents a double-scaling bug.
- **Threshold optimization**: Optimal per-model thresholds (max F1 subject to recall ≥ 0.95) saved as CSV and reused in adversarial evaluation.
- **GNSS constraint enforcement**: Adversarial perturbations clipped to μ ± 6σ bounds from training data.

## Requirements

Python 3.9+, PyTorch, scikit-learn, XGBoost, LightGBM, imbalanced-learn, ART, matplotlib, seaborn, pandas, numpy.
