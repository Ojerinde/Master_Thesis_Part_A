# Kaggle GPU runbooks

Two independent Kaggle notebooks, both driven by the same corpus dataset
(`texbat_track_combined.csv`). Run `kaggle_core_pipeline.ipynb` first if the
locally saved models are stale relative to `data/loader.py`'s scaler (check
`data/processed/scaler.pkl`'s timestamp against the model files under
`results/models/`); `kaggle_generalization.ipynb` is independent and can run
before, after, or in parallel.

## `kaggle_core_pipeline.ipynb` — the 13-detector training + attack suite

Runs `run_pipeline.py` (01 through 22): trains all 13 detectors from scratch,
then the full attack suite (decision-based boundary, multi-surrogate transfer,
FGSM/PGD/DLSA/SNA/TPA) and stats. This is what produces the tables Table 1 and
the fragility/rank-inversion/trade-off figures are built from; run
`papers/paper1-satnav/make_figures.py` locally afterward to regenerate the
figures themselves (`run_pipeline.py` does not call it). Needed whenever the feature
space or the training code changes, since every downstream script assumes the
saved models match the current scaler — running attacks against a model
trained under a different scaler produces silently wrong numbers, not an
error. Much lighter than experiment 23 below (trains each detector once, not
per fold), so it fits comfortably in one Kaggle session.

## `kaggle_generalization.ipynb` — experiment 23 (full generalization)

Purpose: run `experiments/23_generalization.py` (the full version, with the six deep
models) on a Kaggle GPU, because the ~84 per-fold DL retrains are slow on a local CPU
(~7-16 h vs ~1 h on a P100). This experiment is self-contained: it loads the corpus
CSV and retrains every fold from scratch, so it does NOT depend on the locally trained
`.joblib`/`.pt` models, and running it on Kaggle does not disturb any other result.

Reproducibility: `23_generalization.py` now fixes all seeds (Python, NumPy, Torch) and
sets `cudnn.deterministic`, so the Kaggle numbers are stable. The classical folds are
deterministic (`random_state=42`) and reproduce the local `--classical-only` numbers
exactly.

Compute-environment note: the manuscript does not currently state a training
environment (CPU vs. GPU) for any result, so there is nothing in the paper to
reconcile regardless of which of these two notebooks trains the core models.
If that changes, state it in Methods.

--------------------------------------------------------------------------------
## One-time setup

1) Push the code to GitHub (data is gitignored, so only code travels):
       git add -A
       git commit -m "Kaggle: seed determinism in 23_generalization + runbook"
       git push origin paper1-experiment
   The repo is https://github.com/Ojerinde/Master_Research.git . The 316 GB of raw
   recordings (data/raw/*) and the 38 MB corpus (data/processed/*) are gitignored and
   will NOT be pushed. Confirm with `git status` that no .bin/.csv is staged.

2) Upload the corpus CSV as a Kaggle Dataset (once):
   - Kaggle -> Datasets -> New Dataset.
   - Upload data/processed/texbat_track_combined.csv (38 MB).
   - Name it e.g. "texbat-track-corpus". This is the only data Kaggle needs.

--------------------------------------------------------------------------------
## Kaggle notebook

Create a new Notebook. In the right-hand panel:
  - Accelerator: GPU (P100, or T4 x2).
  - Internet: ON (needed to git clone).
  - Add input: the "texbat-track-corpus" dataset.

Paste these cells in order.

### Cell 1 — clone the code
```python
import os, subprocess, sys
REPO = "https://github.com/Ojerinde/Master_Research.git"
BRANCH = "paper1-experiment"          # the Paper-1 branch
DST = "/kaggle/working/repo"
if not os.path.exists(DST):
    subprocess.run(["git","clone","--depth","1","-b",BRANCH,REPO,DST], check=True)
os.chdir(DST)
print("cwd:", os.getcwd()); print(sorted(os.listdir(".")))
# If the repo is PRIVATE, clone with a token instead:
#   REPO = "https://<GITHUB_TOKEN>@github.com/Ojerinde/Master_Research.git"
```

### Cell 2 — place the corpus CSV where the loader expects it
```python
import glob, shutil, os
src = glob.glob("/kaggle/input/**/texbat_track_combined.csv", recursive=True)
assert src, "Attach the dataset that contains texbat_track_combined.csv"
os.makedirs("data/processed", exist_ok=True)
shutil.copy(src[0], "data/processed/texbat_track_combined.csv")
print("CSV placed:", os.path.getsize("data/processed/texbat_track_combined.csv"), "bytes")
```

### Cell 3 — environment check (do NOT pip install -r requirements.txt)
```python
import sys, subprocess, importlib
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
# Kaggle already ships torch/sklearn/xgboost/lightgbm/imbalanced-learn. Install a
# package ONLY if the import fails (requirements.txt is TF-based and must be avoided).
for mod, pip_name in [("imblearn","imbalanced-learn"),("xgboost","xgboost"),("lightgbm","lightgbm")]:
    try: importlib.import_module(mod)
    except ImportError: subprocess.run([sys.executable,"-m","pip","install","-q",pip_name], check=True)
print("deps OK")
```

### Cell 4 — run the full generalization (this is the ~1 h GPU job)
```python
import os, subprocess, sys
env = dict(os.environ, PYTHONPATH="."); env["PYTHONWARNINGS"]="ignore"
# streams per-fold progress; expect cross-scenario (ds2/ds3/ds7) + leave-PRN folds
subprocess.run([sys.executable, "experiments/23_generalization.py"], env=env, check=True)
```

### Cell 5 — save the result for download + quick sanity print
```python
import shutil, pandas as pd
shutil.copy("results/tables/generalization.csv", "/kaggle/working/generalization.csv")
g = pd.read_csv("/kaggle/working/generalization.csv")
print("rows:", len(g), "| families:", g.family.unique().tolist())
print(g.groupby(["protocol","family"])[["recall","f1"]].mean().round(4).to_string())
```

Then download `generalization.csv` from the notebook's Output tab (or "Save Version"
to persist it), and send it back. It should now contain BOTH `classical` and `deep`
rows for `cross_scenario` and `leave_prn`.

--------------------------------------------------------------------------------
## Sanity check on return
- `family` column has both `classical` and `deep`.
- Classical `cross_scenario` mean recall/F1 match the local run (~0.52 / ~0.64).
- Deep folds present for all six models across ds2/ds3/ds7 and the 11 held-out PRNs.
- ds2 (overpowered) remains the collapse case for the deep models too (expected).
