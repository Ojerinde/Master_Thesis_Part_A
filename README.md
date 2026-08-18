# GNSS Adversarial Machine Learning Research

Adversarial robustness analysis of classical and deep learning GNSS spoofing
detectors on receiver observables reprocessed from TEXBAT through the FGI-GSRx
software-defined receiver. Backs the Paper-1 manuscript (`papers/paper1-satnav`
in the workspace root, `paper1-manuscript` branch of this repo; this code is
`main`, corrected 2026-08-18 to match Part_B's branch layout).

## Quick Start

```bash
# Install dependencies (see requirements.txt header — do NOT run this on Kaggle,
# see KAGGLE_RUNBOOK.md)
pip install -r requirements.txt

# Run the current pipeline end to end (trains everything, then attacks + stats + figures)
python run_pipeline.py

# Run the heavy generalization / defense experiments (Kaggle GPU) separately
python run_gpu_experiments.py
```

`main.py` (stages 1-4, `python main.py --stage 1 2`) is the **legacy** entry
point from an earlier StandardScaler-era version of the pipeline. It still
runs, but it is not what produced the reported manuscript numbers. Use
`run_pipeline.py`.

## Pipeline (`run_pipeline.py`)

Runs in dependency order and stops on first failure:

| Script | Description |
| --- | --- |
| `01_classical_baseline.py` | Train 17 classical model variants (7 base families × default/balanced/SMOTE), 5-fold CV, threshold optimization. Pipelines scale internally with `MinMaxScaler`. |
| `02_deep_learning_baseline.py` | Train 6 DL architectures (CNN-1D, LSTM, BiLSTM, CNN-LSTM, Transformer, TCN), 3 seeds each, on the same `MinMaxScaler`-transformed input. |
| `12_operating_point.py` | Common recall-0.95 operating point (`η*`) per detector on the test split. |
| `13_blackbox_attacks.py` | The headline result: a vectorized, gradient-free decision-based boundary attack (Brendel et al. 2018) in min-max `[0,1]` space, applied to all 13 detectors. |
| `14_multisurrogate_transfer.py` | Multi-surrogate PGD transfer attack against the classical models. |
| `18_full_eval.py` | FGSM/PGD/DLSA/SNA/TPA at fixed budgets `{0.05, 0.10, 0.20}`, all in the same min-max `[0,1]` space, at `η*`. |
| `17_latency.py`, `20_mono.py` | Inference/attack-generation latency; PGD monotonicity diagnostic. |
| `19_final_analysis.py`, `22_mcnemar.py` | Bootstrap CIs, worst-case robustness, McNemar pairwise comparability. |
| `21_figures.py`, `16_revision_figures.py` | Data-driven manuscript figures. |

Two experiments run separately (long, GPU-bound; see `KAGGLE_RUNBOOK.md` and
`kaggle_generalization.ipynb`), orchestrated by `run_gpu_experiments.py`:

| Script | Description |
| --- | --- |
| `23_generalization.py` | Cross-scenario (leave-one-scenario-out) and leave-PRN generalization, all 13 models, retrained per fold. |
| `24_defense.py` | Diagnostic adversarial-training baseline (6 DL models) — Paper-2 track, not part of the Paper-1 result set. |

`experiments/03_adversarial_evaluation.py`, `04_statistical_analysis.py`,
`05_manuscript_figures.py` are the **legacy** (pre-min-max) versions, kept for
reference; not part of `run_pipeline.py`.

## Dataset

The operative corpus is `data/processed/texbat_track_combined.csv`: **209,000
epochs, 51.2% spoofed**, from the static TEXBAT family (`cleanStatic`, `ds2`,
`ds3`, `ds7`) reprocessed through FGI-GSRx (see `fgi/export_texbat_track.m`).
Labelling is within-recording (before/after the documented spoofing onset), so
there is no recording-identity shortcut. Every detector consumes the same
**9 physically independent observables** (`data/loader.py::TEXBAT_TRACK_FEATURES`):
`cn0_dbhz, mean_cn0_dbhz, noise_cn0, doppler_hz, i_prompt, q_prompt, dll_discr,
pll_lock, fll_lock`. Derived quantities (carrier frequency = Doppler exactly,
prompt power = I²+Q² exactly, prompt phase = atan2(Q,I) exactly) are excluded
so a feature-space attack cannot produce a physically impossible receiver
state.

The older `data/raw/texbat_channel_combined.csv` (feature-level, ~94,900
samples, 23 engineered features, 33.6/66.4 class split) backed the pre-pivot,
feature-level version of this project and is not used by the current pipeline.

## Splitting

`data/loader.py::load_track_splits` is the single leakage-free split used
everywhere: within each (scenario, satellite, segment) group, epochs are
ordered by time and cut into contiguous 70/10/20% blocks, with a purge gap
dropped at each boundary (adjacent epochs of the same satellite are
near-duplicates through temporal autocorrelation, so a random split would
leak). It returns a `MinMaxScaler` fitted on the training block — this is the
**one shared feature space** every detector and every attack budget is
defined in, so an ε value is directly comparable across models and matches
the decision-based attack's own min-max L∞ measure.

## Project Structure

```
gnss_adversarial_research/
├── run_pipeline.py           # current pipeline entry point
├── run_gpu_experiments.py    # 23_generalization + 24_defense (Kaggle)
├── main.py                   # legacy entry point (stages 1-4)
├── config/                   # paths, model configs, attack configs
├── data/
│   ├── processed/            # texbat_track_combined.csv (operative corpus)
│   ├── raw/                  # legacy texbat_channel_combined.csv
│   └── loader.py             # load_track_splits (the shared MinMaxScaler split)
├── models/
│   ├── classical/             # RandomForest, GradientBoosting, XGBoost, LightGBM, KNN, MLP, DecisionTree
│   └── deep_learning/         # CNN-1D, LSTM, BiLSTM, CNN-LSTM, Transformer, TCN
├── attacks/
│   ├── fgsm.py, pgd.py        # white-box gradient attacks (DL only)
│   └── gnss_attacks.py        # DLSA, SNA, TPA (GNSS domain attacks, cite An et al. 2025 for DLSA/SNA)
├── experiments/                # 01,02,12,13,14,16,17,18,19,20,21,22,23,24 (current); 03,04,05 (legacy)
├── results/
│   ├── models/, tables/, figures/, checkpoints/
└── utils/gnss_constraints.py  # GNSSConstraintEnforcer
```

## Key design decisions

- **One feature space.** Every model and every attack (fixed-budget and
  decision-based) operates in the same min-max `[0,1]` space fitted on the
  training block, so an ε budget means the same thing everywhere.
- **Leakage-free split.** Block-temporal, per (scenario, satellite, segment),
  with a purge gap — see Splitting above.
- **Physical realizability.** `GNSSConstraintEnforcer` clips adversarial
  features to a data-driven `[mean - 6σ, mean + 6σ]` box, and separately
  enforces the physical coupling C/N0 ~ a + b·log10(I²+Q²) (fit on unscaled
  training data), so an attack cannot report a signal-quality value the
  correlator power could not have produced. The three domain attacks
  (DLSA/SNA/TPA) re-project onto their L∞ budget after the enforcer, so
  outlier clipping cannot silently inflate the realized perturbation beyond ε.
- **Gradient-free attack for non-differentiable models.** Tree/KNN detectors
  deny a useful gradient; `13_blackbox_attacks.py` attacks them (and every
  other detector) through a vectorized decision-based boundary search instead
  of relying on transferred gradients, which is what exposes gradient masking.
- **No ART / cleverhans.** The decision-based attack is a custom vectorized
  search (`13_blackbox_attacks.py`); neither library is a runtime dependency.
- **SVM / logistic regression are trained but not in the 13-detector
  adversarial roster.** `01_classical_baseline.py` also fits a logistic
  regression baseline; on this corpus it scores F1 ≈ 0.49-0.51 (see
  `results/tables/baseline_results.csv`), far below every other classical
  model (next-worst is KNN at 0.71), consistent with the feature space not
  being linearly separable. A linear model that barely detects the attack
  is not a meaningful adversarial-robustness test, so it (and SVM) are
  excluded from the 7 classical + 6 deep roster actually attacked.

## Requirements

See `requirements.txt` (PyTorch, scikit-learn, XGBoost, LightGBM,
imbalanced-learn, matplotlib, pandas, numpy; pinned versions). Do not
`pip install -r requirements.txt` on Kaggle — see `KAGGLE_RUNBOOK.md`.
