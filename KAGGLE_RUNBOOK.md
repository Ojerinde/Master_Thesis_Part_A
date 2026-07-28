# Kaggle GPU runbook

Three Kaggle notebooks, all driven by the same corpus dataset
(`texbat_track_combined.csv`). Two of the three are **twins run head-to-head**
for a direct A/B on the Transformer architecture (`kaggle_core_pipeline_bn_only.ipynb`
and `kaggle_core_pipeline_bn_tok.ipynb`) — run both, in parallel on two separate
Kaggle accounts, and compare the resulting Transformer numbers before deciding
which goes in the paper. `kaggle_generalization.ipynb` is independent of both
and can run before, after, or in parallel with either.

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

## `kaggle_core_pipeline_bn_only.ipynb` / `kaggle_core_pipeline_bn_tok.ipynb` — the 14-detector training + attack suite

Two copies of the same notebook, differing in exactly one environment
variable (`TRANSFORMER_MODE=bn_only` vs `TRANSFORMER_MODE=bn_tok`, read by
`models/deep_learning/transformer.py`). Both run `run_pipeline.py` (01, 02,
then 12/13/25/14/18/17/20/19/22 in that order): trains all 14 detectors (7
tree/instance/shallow-NN classical + RBF-kernel SVM + 6 deep) from scratch,
then the full attack suite (decision-based boundary, its bootstrap CI,
multi-surrogate transfer, FGSM/PGD/DLSA/SNA/TPA) and stats.

**Why two variants**: the Transformer collapsed to a constant predictor
(AUC=0.5) under `MinMaxScaler` — root cause was a missing input-normalization
layer that every *other* deep architecture in this codebase already had, now
fixed with `BatchNorm1d` (verified empirically: AUC 0.50 -> 0.78 on real data,
20-epoch capped test). Separately, the model's own docstring always intended
"each feature treated as its own token" for genuine cross-feature attention,
which the code never actually did (all 9 features were collapsed into a
single token, making self-attention a mathematical no-op). Fixing that
(`bn_tok`, per-feature tokenization, FT-Transformer-style) is architecturally
correct but was NOT clearly better empirically in the capped test (`bn_only`
AUC=0.778/F1=0.622 vs `bn_tok` AUC=0.744/F1=0.605, `bn_tok` also ~3.7x slower
per epoch) — inconclusive enough that both need a full-scale run before
deciding which the paper reports.

Why SVM: the direct comparator paper (An et al. 2025) attacks an RBF-kernel
SVM specifically, and SVM is independently well-established in the GNSS
spoofing-detection literature (chen2022svm, Zhu et al. 2022, Aissou et al.
2022) — a linear kernel already failed here for the same reason
LogisticRegression did, so this must be RBF to be a meaningful comparator.
Uses cuML (GPU, exact kernel) if RAPIDS is present in the Kaggle image,
falling back to a scalable CPU approximation (random Fourier features +
linear classifier, Rahimi & Recht 2007) otherwise — plain sklearn `SVC`
would be impractically slow at 144,900 training rows.

This produces the tables Table 1 and the fragility/rank-inversion/trade-off
figures are built from; run `papers/paper1-satnav/make_figures.py` locally
afterward to regenerate the figures themselves (`run_pipeline.py` does not
call it). Needed whenever the feature space or the training code changes,
since every downstream script assumes the saved models match the current
scaler — running attacks against a model trained under a different scaler
produces silently wrong numbers, not an error. Much lighter than experiment
23 below (trains each detector once, not per fold), so each fits comfortably
in one Kaggle session.

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

3) Open the notebooks directly on Kaggle (upload the .ipynb file, or paste
   its cells into a new Kaggle notebook) rather than retyping cells by hand —
   all three are maintained files in this repo, so they are always the
   current version. Run `kaggle_core_pipeline_bn_only.ipynb` and
   `kaggle_core_pipeline_bn_tok.ipynb` on two *separate* Kaggle accounts
   simultaneously (each needs its own `GITHUB_TOKEN` secret attached) to stay
   under any single account's weekly GPU-hour cap.

In the right-hand panel before running any of them: Accelerator = GPU,
Internet = On, Add Input = the corpus dataset ("Add Input", not "New
Dataset" — you are attaching the dataset you already uploaded, not creating
a new one).

--------------------------------------------------------------------------------
## After the Kaggle runs: path to the final paper

Everything below happens locally, after downloading each notebook's Output.

### 1. Unpack both twin runs and decide bn_only vs bn_tok
- Download `results_minmax_bn_only.zip` + `pipeline_log_bn_only.txt` from one
  run, `results_minmax_bn_tok.zip` + `pipeline_log_bn_tok.txt` from the
  other. Unzip each into a **separate** local folder first (don't extract
  either straight into `code/gnss_adversarial_research/results/` yet) so
  they can't clobber each other before you've compared them.
- Compare the Transformer row in each run's `operating_point_recall95.csv`
  and `blackbox_boundary_ci.csv` (both notebooks print this row directly at
  the end of their step 5). Pick whichever is going in the paper — this is a
  judgment call between architectural correctness (`bn_tok`: genuine
  cross-feature attention, matches the model's own documented design intent)
  and empirical performance (whichever scores higher end-to-end; the capped
  diagnostic favoured `bn_only`, full-scale training may not).
- Once decided, extract *only that run's* `results_minmax_bn_*.zip` over
  `code/gnss_adversarial_research/results/` (overwrites the stale pre-rerun
  `models/` and `tables/`). Keep its `pipeline_log_bn_*.txt` — it is the only
  record of the Friedman chi-squared statistics and the McNemar
  significant-pairs count (printed by `19_final_analysis.py` /
  `22_mcnemar.py`, never written to a CSV).
- Note the decision and why in the methods text (see step 4 below) —
  reviewers may ask why one variant was chosen over the other.

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
  ones most likely to move: fig06 (fragility ranking + CI bars), fig08
  (generalization heatmap + leave-PRN strip), fig10 (rank inversion), fig11
  (trade-off parallel-coordinates) — trees (RandomForest/XGBoost/LightGBM/
  GradientBoosting/DecisionTree) are scale-invariant so their relative
  ordering is unlikely to move much; KNN, MLP, and all six deep models can
  shift meaningfully since they were retrained under a different scaler.

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
    generalization headline sentences).
  - `sections/introduction.tex` — the four-result summary paragraph, mirrors
    the abstract.
  - `sections/results.tex` — almost every number in this file: Table 1
    (`tab:oppoint`), the fragility paragraph (medians + CIs + the tree-cluster
    overlap claim), the domain-ASR paragraph, the generalization paragraph
    (cross-scenario means/sd, leave-PRN medians), the McNemar/Friedman
    paragraph (from the chosen run's `pipeline_log_bn_*.txt`, not
    `verify_numbers.py`).
  - `sections/discussion.tex` — the FAR range in the operational-synthesis
    paragraph (`0.30 to 0.69`, from Table 1).
  - `sections/threat_model.tex` — the epsilon-to-dB calibration paragraph
    cites the most/least fragile detector's specific values (currently "near
    0.01 ... falls to ... 0.5 dB" / "about 0.10, roughly 6 dB"); re-check
    these are still GradientBoosting/KNN and the dB figures still hold once
    the fragility ranking is refreshed.
  - `sections/methods.tex` — **every "thirteen detectors" becomes "fourteen
    detectors"** (abstract, intro contributions, methods, every figure/table
    caption that states the count) now that RBF-kernel SVM is in the roster.
    The linear-classifier exclusion sentence is currently qualitative on
    purpose ("far below every model", no number given) because this exact
    rerun was expected to change it — fill in the real LogisticRegression F1
    from `results/tables/baseline_results.csv` now that it is fresh. Also add
    a sentence on which Transformer variant was used and why (step 1 above) —
    if `bn_tok`, note the per-feature tokenization/genuine-attention design;
    if `bn_only`, be honest that self-attention reduces to a no-op at
    seq_len=1 and the model is functionally a residual MLP, since a careful
    reader could reconstruct this from the code either way.

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
  scenario. If a retrained model's ranking moved enough to change which
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
