# Kaggle GPU runbook

Three Kaggle notebooks, all driven by the same corpus dataset
(`texbat_track_combined.csv`):

- `kaggle_core_pipeline.ipynb` — full from-scratch retrain of all 14
  detectors + the attack suite. Only needed if the corpus or the training
  code changes; **not** needed just to pick up the SVM tuning fix below.
- `kaggle_svm_retrain.ipynb` — retrains **only SVM**, reusing the other 13
  detectors from a prior full run. This is the one to use right now: the
  original full run's SVM entry never got hyperparameter tuning (a grid was
  missing for it), and this notebook fixes that without repeating the ~6-7h
  of already-correct training for the other 13 detectors.
- `kaggle_generalization.ipynb` — experiment 23, independent of both, can run
  before/after/in parallel with either.

Repo: **https://github.com/Ojerinde/Master_Thesis_Part_A** (`paper1-experiment`
branch). Note this repo also holds the manuscript itself, on `main` —
`paper1-experiment` is a separate branch and pushing to it never touches
`main`. The old `Master_Research` repo referenced in earlier versions of this
file was deleted; this is the current one.

**This repo is private.** All three notebooks' clone cell reads a `GITHUB_TOKEN`
from Kaggle Secrets (Add-ons -> Secrets in the notebook's right panel) rather
than an anonymous clone, which fails with `could not read Username`. Create a
fine-grained GitHub PAT scoped to only this repo with `Contents: Read-only`,
add it as a Kaggle Secret named `GITHUB_TOKEN`, and attach it to each
notebook that needs to clone (per-notebook, not shared). Never paste the
token directly into a cell — Kaggle notebooks can end up shared or public,
Secrets cannot.

## `kaggle_core_pipeline.ipynb` — the 14-detector training + attack suite (full retrain)

Runs `run_pipeline.py` (01, 02, then 12/13/25/14/18/17/20/19/22 in that
order): trains all 14 detectors (7 tree/instance/shallow-NN classical + tuned
RBF-kernel SVM + 6 deep) from scratch, then the full attack suite
(decision-based boundary, its bootstrap CI, multi-surrogate transfer,
FGSM/PGD/DLSA/SNA/TPA) and stats.

**Transformer architecture (settled, not a live A/B anymore):** the
Transformer collapsed to a constant predictor (AUC=0.5) under `MinMaxScaler`
— root cause was a missing input-normalization layer that every *other* deep
architecture in this codebase already had, fixed with `BatchNorm1d`.
Separately, the model's own docstring always intended "each feature treated
as its own token" for genuine cross-feature attention, which the code never
actually did (all 9 features were collapsed into a single token, making
self-attention a mathematical no-op). Both a per-feature-tokenization variant
(`bn_tok`, genuine attention) and a single-token variant (`bn_only`,
attention reduced to a no-op — functionally a residual MLP) were run
head-to-head at full scale; performance was statistically indistinguishable
(overlapping 95% fragility CIs), so the choice came down to which one is
actually doing what "Transformer" means. `bn_tok` (per-feature tokenization)
is what's now hardcoded in `models/deep_learning/transformer.py` — there is
no more `TRANSFORMER_MODE` environment variable and no more twin-notebook
A/B; that decision is final.

**SVM tuning (fixed):** SVM now gets the same 5-fold-CV GridSearchCV tuning
(`C`, `gamma`) as every other classical model — the original run trained it
at its config default only, because `HYPERPARAM_GRIDS` was missing an entry
for it. Uses cuML (GPU, exact kernel) if RAPIDS is present in the Kaggle
image, falling back to a scalable CPU approximation (random Fourier features
+ linear classifier, Rahimi & Recht 2007) otherwise — plain sklearn `SVC`
would be impractically slow at 144,900 training rows, and the CPU fallback's
differently-shaped Pipeline doesn't support the same tuning grid, so it skips
tuning and trains at the config default (logged clearly, not silent). SVM is
independently well-established in the GNSS spoofing-detection literature
(chen2022svm, Zhu et al. 2022, Aissou et al. 2022) and is the family the
direct comparator paper (An et al. 2025) attacks — a linear kernel already
failed here for the same reason LogisticRegression did, so this must be RBF
to be a meaningful comparator.

This produces the tables Table 1 and the fragility/rank-inversion/trade-off
figures are built from; run `papers/paper1-satnav/make_figures.py` locally
afterward to regenerate the figures themselves (`run_pipeline.py` does not
call it). Needed whenever the feature space or the training code changes,
since every downstream script assumes the saved models match the current
scaler — running attacks against a model trained under a different scaler
produces silently wrong numbers, not an error. Much lighter than experiment
23 below (trains each detector once, not per fold), so it fits comfortably
in one Kaggle session.

## `kaggle_svm_retrain.ipynb` — retrain ONLY SVM, reuse the other 13 detectors

For the common case where only SVM needs retraining (e.g. picking up a
tuning fix or config change) and the other 13 detectors from a prior full run
are still valid. Runs `run_pipeline_svm_retrain.py`:
`01b_retrain_svm_only -> 12 -> 13 -> 25 -> 14 -> 18 -> 17 -> 20 -> 19 -> 22`
— the same downstream stages as the full pipeline, but skipping stage 01's
GridSearchCV over the other 17 already-correctly-tuned classical variants and
skipping stage 02's DL training entirely.

**Requires a second Kaggle Dataset input** beyond the corpus CSV: the prior
run's `results/models/` folder (7 other classical `.joblib` + 6 deep
learning `.pt` files), so the downstream stages have something to reload
for the 13 detectors this notebook doesn't retrain. The notebook's Step 1c
fails loudly, listing exactly what's missing, if this isn't attached — the
downstream stages themselves only print "MISSING <name>" and silently
continue otherwise, which is exactly the kind of quietly-incomplete run this
preflight check exists to prevent (see `run_pipeline_svm_retrain.py`'s
`preflight()`).

`experiments/01b_retrain_svm_only.py` reuses `01_classical_baseline.py`'s
training/calibration/evaluation code directly (via `importlib`, since a
filename starting with a digit can't be `import`-ed normally) rather than
duplicating it, so there is no risk of the two drifting apart. It reproduces
the identical deterministic block-temporal train/val/test split (no RNG in
`data.loader.load_track_splits`, so every independent call/process gets the
same partition — the same assumption every other stage already relies on),
trains and tunes only the SVM Pipeline, and overwrites only `SVM.joblib` plus
merges its row into the two `01`-owned diagnostic CSVs.

One thing this notebook cannot verify: that the models you upload in Step 1c
were actually trained under the *current* code. A stale `transformer.pt` from
before the `BatchNorm1d`/tokenizer fix, for instance, will fail to load with
a clear `RuntimeError` (missing/unexpected state_dict keys) at stage 12 —
loud and immediate, not silently wrong, but still worth a moment's sanity
check on which run you're uploading.

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

1) Push the code (data is gitignored, so only code travels):
       git add -A
       git commit -m "..."
       git push origin paper1-experiment
   The 316 GB of raw recordings (data/raw/*) and the 38 MB corpus
   (data/processed/*) are gitignored and will NOT be pushed. Confirm with
   `git status` that no .bin/.csv is staged.

2) Upload the corpus CSV as a Kaggle Dataset (once, already done as of this
   writing — re-use the existing dataset, do not re-upload):
   - Kaggle -> Datasets -> New Dataset.
   - Upload data/processed/texbat_track_combined.csv (38 MB).
   - Name it e.g. "texbat-track-corpus".

2b) Only for `kaggle_svm_retrain.ipynb`: also upload the prior full run's
   `results/models/` folder as a second Kaggle Dataset (e.g. from your local
   unzipped `results_minmax_bn_tok/models/`). See that notebook's Step 1c for
   exactly what's expected. Re-upload a new version of this dataset any time
   you have a newer "other 13 detectors" baseline to retrain SVM against.

3) Open the notebooks directly on Kaggle (upload the .ipynb file, or paste
   its cells into a new Kaggle notebook) rather than retyping cells by hand —
   all three are maintained files in this repo, so they are always the
   current version.

In the right-hand panel before running any of them: Accelerator = GPU,
Internet = On, Add Input = the corpus dataset (and, for
`kaggle_svm_retrain.ipynb`, the prior-run models dataset too) — "Add Input",
not "New Dataset" — you are attaching datasets you already uploaded, not
creating new ones.

--------------------------------------------------------------------------------
## After the Kaggle runs: path to the final paper

Everything below happens locally, after downloading each notebook's Output.

### 1. Unpack the SVM retrain over the prior full run
- Download `results_svm_retrain.zip` + `pipeline_log_svm_retrain.txt` from
  `kaggle_svm_retrain.ipynb`'s Output tab.
- Extract it over `code/gnss_adversarial_research/results/` (overwrites
  `SVM.joblib` and every table with the tuned-SVM numbers; the other 13
  detectors' files are already identical since the notebook reused them
  as-is). Keep `pipeline_log_svm_retrain.txt` — it is the only record of the
  Friedman chi-squared statistics and the McNemar significant-pairs count
  (printed by `19_final_analysis.py` / `22_mcnemar.py`, never written to a
  CSV).
- If instead you ran a full `kaggle_core_pipeline.ipynb` retrain (corpus or
  training code changed), extract `results_minmax.zip` the same way and keep
  `pipeline_log.txt` instead.

### 2. Run `kaggle_generalization.ipynb` (experiment 23)
- Independent of step 1; can happen before, after, or already be done.
- Kaggle's 12h session cap does not preserve output on a timeout, so this
  runs in chunks across possibly several commits (`--protocol cross_scenario`
  first, then `--protocol leave_prn` in batches of a few PRNs, reseeding the
  previous partial `generalization.csv` each time — see the notebook's own
  cell 2b). Keep going until the sanity check at the bottom of this file
  passes (both `classical` and `deep` rows, all 11 PRNs present).
- Download the final `generalization.csv` and drop it into
  `code/gnss_adversarial_research/results/tables/generalization.csv`.

### 3. Regenerate the manuscript figures
       cd "papers/paper1-satnav"
       python make_figures.py
  Regenerates all 11 figures from the fresh tables in
  `code/gnss_adversarial_research/results/tables/`. Visually spot-check the
  ones most likely to move: fig06 (fragility ranking + CI bars), fig10 (rank
  inversion), fig11 (trade-off parallel-coordinates) — SVM is the only
  detector whose training changed this round, so anything not involving SVM
  should be pixel-identical to the prior `bn_tok` run; fig08 (generalization
  heatmap + leave-PRN strip) only moves if step 2 produced new numbers.

### 4. Recompute every number the manuscript cites
       cd "papers/paper1-satnav"
       python verify_numbers.py
  Reads directly from the fresh `results/tables/` and prints the source-of-
  truth value for every number the manuscript uses: clean detection (Table
  1), both operating points, fragility median + 95% CI (including the
  tree-cluster overlap claim), domain-attack ASR, generalization, latency.
  Cross-check its output against these files, in order of how much text
  depends on retrained models:
  - `main.tex` — abstract (`F1 between 0.74 and 0.84`, the fragility/
    generalization headline sentences) — recheck the range still holds now
    that SVM's F1 moved.
  - `sections/introduction.tex` — the four-result summary paragraph, mirrors
    the abstract.
  - `sections/results.tex` — almost every number in this file: Table 1
    (`tab:oppoint`), the fragility paragraph (medians + CIs + the tree-cluster
    overlap claim — SVM was already the weakest detector pre-tuning, recheck
    it doesn't move enough to change the ranking claims), the domain-ASR
    paragraph, the generalization paragraph (cross-scenario means/sd,
    leave-PRN medians), the McNemar/Friedman paragraph (from
    `pipeline_log_svm_retrain.txt`, not `verify_numbers.py`).
  - `sections/discussion.tex` — the FAR range in the operational-synthesis
    paragraph (`0.30 to 0.69`, from Table 1) — SVM had the worst FAR
    pre-tuning (~0.94), recheck whether tuning moved the range's endpoint.
  - `sections/threat_model.tex` — the epsilon-to-dB calibration paragraph
    cites the most/least fragile detector's specific values (currently "near
    0.01 ... falls to ... 0.5 dB" / "about 0.10, roughly 6 dB"); re-check
    these are still GradientBoosting/KNN once the fragility ranking is
    refreshed (SVM was not the extreme on either end pre-tuning, but confirm).
  - `sections/methods.tex` — confirm "fourteen detectors" (not thirteen) is
    used consistently (abstract, intro contributions, methods, every
    figure/table caption that states the count) now that RBF-kernel SVM is
    in the roster. Fill in the real LogisticRegression F1 from
    `results/tables/baseline_results.csv` if the linear-classifier exclusion
    sentence is still qualitative-only. State the Transformer architecture
    (per-feature tokenization, genuine attention — see the settled-decision
    note above) and that SVM was tuned via GridSearchCV like every other
    classical model.

### 5. Rebuild and verify the PDF
       pdflatex -interaction=nonstopmode main && bibtex main && pdflatex -interaction=nonstopmode main && pdflatex -interaction=nonstopmode main
  Confirm: 0 undefined citations/references, 0 errors, 0 overfull hboxes
  greater than 2pt. This has been the bar for every revision so far (last
  clean build: 25 pages).

### 6. Read the results section once more, end to end
  Not a number check — a narrative check. Confirm the headline claims still
  hold qualitatively, not just numerically: no family dominates on clean
  data; the decision-based attack still drives every detector to near-zero
  worst-case recall; the tree/nearest-neighbour "robustness" under transfer
  is still gradient masking (i.e. still fragile under the boundary attack);
  cross-scenario generalization still collapses on the unseen overpowered
  scenario. If SVM's tuning moved its ranking enough to change which
  detector is "most fragile" or "the robust outlier", the prose naming that
  detector needs updating too, not just the number next to it.

### Still open, not part of this rerun
- Title change — your call, not automated by anything above.
- Fourth author (Beihang faculty) — TBC, not blocking.
- `_prerevision_backup_2026-07-03/` (694 MB, untracked, permanent if deleted)
  — your call.

--------------------------------------------------------------------------------
## Sanity check on `generalization.csv` specifically
- `family` column has both `classical` and `deep`.
- Classical `cross_scenario` mean recall/F1 in the same neighbourhood as the
  pre-rerun local numbers (~0.52 / ~0.64) — expect some movement, not a
  collapse to near-zero or near-one, which would signal a real bug.
- Deep folds present for all six models across ds2/ds3/ds7 and the 11 held-out PRNs.
- ds2 (overpowered) remains the collapse case for the deep models too (expected).
