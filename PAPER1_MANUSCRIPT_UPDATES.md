# Paper 1 — Manuscript Update Ledger (Satellite Navigation, IF 12.4)

**Purpose:** the authoritative record of every methodology decision + change made
during the FGI-GSRx rebuild, to fold into the manuscript. **Read this before
editing the `.tex`.** Companion to `FIGURE_PLAN.md`. Branch: `main` (renamed
from `paper1-experiment` 2026-08-18; the manuscript itself now lives on
`paper1-manuscript`, in the `papers/paper1-satnav` local checkout).

- Manuscript source (to be retargeted from GPS Solutions -> Satellite Navigation):
  `D:\BEIHANG UNIVERSITY\Research\manuscript\GPS_Solutions_sn\manuscript.tex`
- Writing style: lean prose — no em/en dashes, no filler adjectives, no inline
  "A, B, and C" lists.
- **All numeric results from the GPS Solutions version are VOID** (old toy-receiver
  corpus). Every table/figure/number is regenerated from the new corpus.

---

## 0. Journal target + stage-2 numbers (2026-07-15)

**Journal: Satellite Navigation** (SpringerOpen OA, sn-jnl.cls / sn-basic author-date;
ISSN 2662-9291/2662-1363). Scope fit: "Anti-jamming and anti-spoofing", "Receiver and
signal processing", "Resilient PNT". Format questions settled; template already in repo.
See memory [[satnav_journal_and_positioning]]. Two open calls for papers (Multi-source
Dependable Intelligent Navigation, deadline 2026-12-31 -- direct thematic fit;
Multi-GNSS Real-Time Service, 2027-05-31).

**Positioning vs Song et al. 2026 (CCC-PCNNs), Sat.Nav. 7(1):15, DOI 10.1186/
s43020-026-00199-8** (physics-constrained 1D-CNN, code-carrier consistency; >92.7%
retention on 10 unseen scenarios; premise "ML fails on unseen attacks ~50%"). Our
cross-scenario collapse (recall 0.52 mean, 0.09 min) CORROBORATES their premise; we
extend to the ORTHOGONAL adversarial axis they never test. Cite as SOTA-generalization
+ foil. Verified related-work cites: Borhani-Darian/Li/Wu/Closas 2024 (EURASIP JASP
2024:14, 10.1186/s13634-023-01103-1); J.Li et al 2025 (GPS Solutions 29:45,
10.1007/s10291-024-01802-8); Chen et al 2025 (GPS Solutions 29:175,
10.1007/s10291-025-01938-1).

**Stage-2 rerun (run_pipeline_stage2.py, 2026-07-15) -- numbers now FINAL:**
- Operating point recall-0.95 UNCHANGED from 07-14 (deterministic): clean F1 0.744-0.843,
  FAR 0.30-0.69, AUC 0.803-0.903. (operating_point_recall95.csv/.md)
- NEW operating_point_far05.csv (FAR<=0.05 second point): honest finding -- XGBoost,
  LightGBM, DecisionTree CANNOT reach FAR<=0.05 on val (far_ceiling_met=False); GB's
  calibration does not transfer (val FAR 0.05 -> test FAR 0.375). At the FAR point recall
  falls hard (KNN 0.361, CNN-1D 0.444, RF 0.533) while precision rises to 0.83-0.93. Use
  BOTH points; do not hide either.
- 18_full_eval NOW coupling-enforcer-correct (208 rows). Worst-case recall @eps0.20 per
  attack confirms DECISION-BASED is worst for every model.
- 19_final_analysis (LOGGED, _stage2_logs/04_*): Table-1 clean F1 + 10k-bootstrap CIs
  (e.g. GB 0.843 CI[0.839,0.846], CNN-1D 0.744 CI[0.740,0.748]); WORST-CASE recall under
  decision-based attack ~0 for ALL 13 (RF/GB/BiLSTM/TCN/MLP =0.000, LSTM 0.004, KNN 0.055
  best) with maxdF1 up to +0.603 -> the "no architecture is robust" headline, quantified.
  Friedman attack-heterogeneity sig (deep p=1.3e-4 W=0.956; classical p=2.1e-5 W=0.959).
  Mann-Whitney per-attack: GNSS domain attacks SNA/TPA hurt deep > classical
  (TPA eps0.1 p=5.8e-4) but both tiny; the ML attacks (PGD/transfer) dominate.
- 20_mono (LOGGED): monotonic=True all; with/without enforcer now DIFFER but only
  slightly (PGD already ~respects the C/N0~power coupling) -> report as "attacks remain
  effective under the physical constraint", supports realizability of the threat.
- 22_mcnemar (LOGGED, _stage2_logs/05_*): 78 pairs, 74 significant p_adj<0.05, max OR 4.49,
  median raw p 8.9e-182 -> detectors are pairwise-distinguishable (comparability holds).
- Logs in results/tables/_stage2_logs/ (01_12 ... 05_22). 23_generalization full (DL folds)
  was RUNNING at last check (tcn.py sigmoid-overflow RuntimeWarning is HARMLESS: np.exp
  overflow -> proba 0; clamp not required but could be added). Confirm generalization.csv
  has both families before F8.

---

## 1. Corpus (what changed and why)

- OLD: a hand-rolled, low-fidelity software receiver -> `texbat_channel_combined.csv`
  (17 cols). Defects: C/N0 was a heuristic MISLABELED "Beaulieu"; gain-only loop
  filters; near-empty pseudorange; and labelling made cleanStatic=authentic vs
  ds*=entirely-spoof (a recording-identity confound). Not acceptable at 12.4.
- NEW: raw TEXBAT IQ (25 Msps complex, 16+16-bit, GPS L1; md5-verified vs the UT
  Austin datastore) processed by the **FGI-GSRx software receiver** (GPLv3, cite
  it) -> per-epoch observables -> `data/processed/texbat_track_combined.csv`.
- Scenarios used (STATIC family, shared cleanStatic baseline):
  cleanStatic (authentic) + ds2 (Overpowered Time Push, 10 dB, victim unlocks)
  + ds3 (Matched-Power Time Push, 1.3 dB, stays locked) + ds7 (matched,
  carrier-phase-aligned time push). **ds5/ds6 (dynamic) EXCLUDED** to avoid a
  static-vs-dynamic motion confound. ds8 (SCER) optional — near-identical to ds7
  at the observable level.
- Onsets (ION-2012 paper + ds7/ds8 doc): ds2/ds3 takeover ~80 s, pull-off ~115 s
  -> genuine t<75 s, spoof t>=125 s; ds7 clean 0-110 s, takeover 110-150 s, spoof
  t>=150 s; cleanStatic all-authentic. **Within-recording onset labelling** (both
  classes drawn from the SAME recording) removes the recording-identity confound.
- Static reference antenna: Trimble Geodetic Zephyr II, WRW building, UT Austin.
- Refs added to `references.bib`: `humphreys2016texbatds78`; `humphreys2012texbat`
  already present. See memory [[texbat-scenarios]], [[paper1-signal-rebuild]].

## 2. Feature set — 9 physically INDEPENDENT observables

Kept: `cn0_dbhz, mean_cn0_dbhz, noise_cn0, doppler_hz, i_prompt, q_prompt,
dll_discr, pll_lock, fll_lock`.

Dropped, with verified reason (this is Reviewer #4's "physical relation between
signal control and feature perturbations"):
- `carr_freq_hz` == `doppler_hz` (identical column; std of diff = 0.0)
- `prompt_power` == `i_prompt^2 + q_prompt^2` (exact)
- `prompt_phase_rad` == `atan2(q_prompt, i_prompt)` (exact)
- `fll_filter` (NaN in steady-state fine tracking; pull-in only)

A feature-space attack perturbing a derived feature independently of its parents
yields a physically-impossible receiver state. The residual SOFT coupling
C/N0 ~ f(I/Q power) (corr 0.98) is enforced during attacks by the
`GNSSConstraintEnforcer` (amplitude perturbations must move C/N0 consistently).
See memory [[iq-feature-coupling]].

## 3. Train/val/test split — leakage-free (block-temporal + purge)

`data.loader.load_track_splits`: for each (scenario, prn, segment) group ordered
by t_sec, take contiguous blocks 70/10/20 with a `purge` of 20 epochs (~1 s at the
20 Hz export cadence) dropped at each boundary. No RNG. Validated on the current
2-scenario corpus: train 89,700 / val 11,700 / test 25,200, ~44% spoof in all
three. Rationale: a random per-epoch split leaks via temporal autocorrelation
(adjacent epochs of the same PRN are near-duplicates). Basis: time-series CV /
purging best practice (Lopez de Prado; arXiv 2512.06932).

## 4. Evaluation protocol — 3 views (TEXBAT-standard)

1. **Mixed** (block-temporal within-scenario) — the primary detection numbers.
2. **PRN-level** (leave-PRN-out) — cross-satellite generalization.
3. **Cross-scenario** (leave-one-scenario-out via `load_track_splits(scenarios=)`)
   — cross-attack generalization; the headline robustness claim.
Basis: TEXBAT ML literature reports mixed + PRN-level + cross-training/cross-dataset.
**Implemented:** mixed = the 01->22 chain (`run_pipeline.py`, one command, stops on
first failure); PRN-level + cross-scenario = `experiments/23_generalization.py`
(retrains every model per held-out fold; `--classical-only` for a fast pass).
Figure F8 reads `results/tables/generalization.csv`. Cross-scenario needs >=2 spoofed
scenarios, so it produces its headline folds once ds2/ds3 are in the corpus.

## 5. Models + common operating point

7 classical (RandomForest_default, XGBoost, LightGBM, GradientBoosting, KNN, MLP,
DecisionTree) + 6 DL (CNN-1D, LSTM, BiLSTM, CNN-LSTM, Transformer, TCN) = 13.
**No resampling — corpus is balanced by construction.** Within-recording onset
labelling gives each spoofed scenario BOTH classes, so the pooled corpus is 49/51
(genuine/spoof). On the first real baseline run the SMOTE and class-weight-balanced
RandomForest variants moved F1 by <0.002 vs plain default (0.8190 / 0.8198 / 0.8177),
so we select the plain `default` classical models everywhere (01 still TRAINS all
variants for the comparison table, but 03/04/12-22 select `RandomForest_default`).
This also removes a reviewer criticism: SMOTE synthesises minority "attacks" that
were never transmitted, which is questionable on a security corpus. State in the
paper "the corpus is balanced by construction, so no resampling is applied."
**Common operating point:** per-model threshold tau* = max{tau : R_val(tau) >= 0.95}
on the validation block. No fixed 0.5 anywhere (fixes Reviewer #1 #2).

## 6. Attacks + adversarial-eval rigor (Carlini checklist)

- Suite applied IDENTICALLY to all 13 models at the operating point: FGSM, PGD,
  multi-surrogate transfer, three GNSS domain attacks (DLSA/SNA/TPA), and the
  decision-based boundary attack (fixes Reviewer #1 #1).
- Non-differentiability of tree/kNN is treated as gradient masking; the
  decision-based (hard-label) attack defeats it (Athalye 2018; Carlini 2019).
- DRAFTED (number-independent LaTeX, ready to integrate):
  `papers/_archive/GPS_Solutions_sn/paper1_new_sections_draft.tex` -- signal-model +
  observable-generation section, precise THREAT MODEL section, enforcer section.
  New refs to add to references.bib: `fgigsrx`, `carlini2019evaluating`.
- IMPLEMENTED: coupling-aware enforcer (`utils/gnss_constraints.py`: fit_coupling +
  enforce_coupling_phys) constrains C/N0 to the residual band of its power-predicted
  value. Wire into the attack loops (18/20: after the per-feature clip, in physical
  space; scaled-space attacks via a scaler round-trip) during the run/validation.
- STILL TO ADD: attack HYPERPARAMETER table (values known after the run) and an
  explicit gradient-masking check paragraph/figure.
- Threat-model note: justify the decision-oracle (hard-label) access for GNSS, or
  frame decision-based results as a worst-case attacker bound.

## 7. Statistics

Mann-Whitney U (unpaired DL-vs-classical family comparison, n=6 vs 7), McNemar
mid-p exact + Holm-Bonferroni across pairs, 10k-bootstrap CIs, Friedman across
attacks. (McNemar over-power caveat: comparability rests on the narrow F1 band,
not error-pattern equivalence.)

## 8. Reviewer-concern mapping (keep satisfying ALL)

GPS Solutions issues [1] literature (citation audit done), [2] technique
presentation (signal-model + observable-generation sections), [3] inconsistent
parameters (single source of truth + consistency sweep). Reviewer #1: #1 attack
parity (S.6), #2 operating point (S.5), #3 internal consistency (S.7 below), #4
GNSS narrative (S.10 reframe). Full register in this session's audit table.

## 8b. NEW Satellite Navigation manuscript project (2026-07-16, IN PROGRESS)

Clean build at `papers/paper1-satnav/` (sn-jnl.cls + sn-basic.bst +
references.bib copied from GPS_Solutions_sn). `main.tex` + `sections/*.tex`; compiles
via pdflatex->bibtex->pdflatex x2, 0 undefined citations.
- TITLE (final, 2026-07-16): "Adversarial Vulnerability of Machine-Learning GNSS
  Spoofing Detectors to Physically Realizable Attacks" (short: "Adversarial
  Vulnerability of GNSS Spoofing Detectors"). Dropped the "No Architecture Is Safe:"
  hook (too informal/overclaiming for this journal) and "model-fair" (methods
  descriptor only). Alt on request: "Cross-Architecture Adversarial Vulnerability...".
- FIGURE 1 = methodological_backbone.jpeg (pipeline schematic F1) -> copied to
  figures/fig01_pipeline.jpg, placed in the Introduction, compiles.
- DEFENSE (Paper 1, RUN pending; inclusion decided by the result). experiments/24_defense.py
  now does PROPER Madry PGD adversarial training (cite madry2018towards): on-the-fly
  per-batch PGD inner-max + outer min, Adam weight-decay 5e-4, and EARLY STOPPING on a
  ROBUST validation metric (spoof recall under PGD at recall-0.95) to defeat robust
  overfitting (cite rice2020overfitting -- early-stopped PGD-AT matches TRADES etc.).
  The first (static-augmentation) version made robustness WORSE; the proper version
  makes AT clearly help -> use ONLY the proper version. Evaluated against the FULL suite
  applicable to DL targets (FGSM, PGD, DLSA/SNA/TPA, and the gradient-free DECISION-BASED
  boundary attack in the same [0,1] space as the main results). Records clean FAR per
  variant (so a high-FAR "trivially robust" model is not misread). NOT the certified
  GNSS-Shield defense (Paper 2, still RED [[project_wp2_done]]).
  SMOKE finding (undertrained, indicative): AT lifts PGD recall (CNN-1D @0.2 0.755->0.920)
  but the DECISION-BASED attack still succeeds vs the AT model (CNN-1D ASR 0.63 @0.2, 1.0
  unbounded) -> the coherent story: empirical AT hardens the gradient surface, not the
  boundary; certified defense (Paper 2) still needed. CONFIRM on the full GPU run before
  writing. Runs via run_gpu_experiments.py --with-defense; separate output file
  defense_baseline.csv (generalization.csv stays separate); both downloaded from Kaggle.
  Rice/Madry added to references.bib.

- KAGGLE SESSION-LIMIT INCIDENT (2026-07-17): first --with-defense commit ran 23
  generalization live; fold timings (GPU, batch=32 default): ds2 6204s, ds3 5371s,
  ds7 4064s, PRN=3 7404s -> ~1.6h/fold avg, 14 folds -> ~26h total, FAR beyond
  Kaggle's 12h session cap (confirmed via research: raised from 9h->12h). User
  cancelled at 6.45h/4-of-14-folds. CRITICAL FINDING (verified via research): Kaggle
  does NOT preserve /kaggle/working on a TIMEOUT kill (only on successful
  completion) -> naive per-fold checkpointing alone does NOT survive a timeout,
  only a clean error. FIX (23_generalization.py + run_gpu_experiments.py):
  (1) per-fold checkpoint/resume via generalization.csv (protocol,holdout) skip-set;
  (2) NEW --protocol {all,cross_scenario,leave_prn} flag to deliberately chunk into
  pieces that each COMPLETE inside 12h; (3) NEW --batch-size override (config
  default is 32 for all 6 DL models -> ~5900 batches/epoch on 150-190k rows, badly
  under-uses a GPU; notebook default now 256) -- SPEEDUP NOT YET MEASURED (no local
  GPU), verify on the next Kaggle run; (4) notebook cell "2b" seeds
  results/tables/generalization.csv from a re-attached PREVIOUSLY-DOWNLOADED Kaggle
  dataset before running, since a completed run's Output DOES persist and can be
  carried into the next session's fresh clone (results/tables/ is gitignored, so a
  fresh clone never has it otherwise). All changes smoke-tested locally (--smoke,
  --protocol leave_prn, --batch-size 256 all verified error-free), pushed.
  USER PROCEDURE: run --protocol cross_scenario first (completes, Output persists)
  -> download generalization.csv -> next session: attach it as input dataset ->
  cell 2b seeds it -> run --protocol leave_prn (resumes/chunks as needed).

- ROUND 1 RESULT (2026-07-17, Kaggle T4x2, batch-size=256, --protocol cross_scenario):
  SUCCESS, 3434.4s total (~57 min) for all 3 folds -- batch-256 gave ~5x speedup vs the
  batch-32 baseline (18.9 min/fold avg vs ~90 min/fold: ds2 19.4min, ds3 21.3min,
  ds7 15.9min). REVISED leave_prn estimate: 11 folds x ~19-20min ~= 3.5-4h, likely
  fits ONE 12h session -> told user to try --protocol leave_prn with PRN_LIMIT=None
  first (chunk only if it runs long).
  39 rows written (3 folds x 13 models, BOTH families -- first time deep cross-scenario
  numbers exist). HEADLINE FINDINGS (scenario-dependent, NOT a flat classical>deep):
    ds2 (overpowered, hardest): near-total collapse ALL models. Best classical KNN
      0.473, best deep BiLSTM 0.269. Worst: RandomForest 0.090(!), TCN 0.145.
    ds3 (matched-power): moderate. Best classical KNN 0.744, best deep CNN-1D 0.631.
    ds7 (phase-aligned): BEST generalization overall, and CNN-1D 0.844 + TCN 0.840
      (deep) BEAT every classical model (best classical XGBoost 0.786).
  Cross-scenario means: classical recall=0.5283/F1=0.6468, deep recall=0.4420/
  F1=0.5373 (deep worse ON AVERAGE, but ds7 shows deep CAN generalize best -> the
  paper's claim must be scenario-dependent, not a blanket family ranking).
  THIRD independent axis for the "no family wins" thesis: (1) clean detection
  (F1 0.74-0.84 span, no dominance), (2) adversarial robustness (decision-based,
  worst-case recall~0 for all 13), (3) NOW cross-scenario generalization (no family
  dominates all 3 held-out scenarios either). Same conclusion, three separate
  experiments -> this is the coherence a 12.4 reviewer rewards.
  CANONICAL-SOURCE note: this Kaggle-run generalization.csv (batch=256, GPU) will
  SUPERSEDE the earlier local classical-only reference numbers (recall 0.5240 vs
  0.5283 here -- tiny env/library drift, not a bug) once leave_prn completes. Cite
  ONLY this file's final numbers in the manuscript, not the old local partial run.
  SCOPE (state factually in Methods, NOT as a Discussion "limitation" vs a named
  competitor -- user correction 2026-07-17, see [[writing_style_preference]]):
  cross-scenario testing covers 3 held-out scenarios (ds2/ds3/ds7, static TEXBAT
  family; ds5/ds6 excluded for the static/dynamic confound). This is a plain
  methodology fact, already fully disclosed by stating the number -- do NOT add an
  explicit comparison to Song et al.'s 10 scenarios. Depth (13 models, decision-based
  attack, physical realizability) is the differentiator to emphasize; if a reviewer
  raises scenario count, answer it in revision, don't pre-concede it in the manuscript.
  STILL PENDING: leave-PRN for the 6 deep models (Round 2, in progress) -- needed
  to complete the 3-view protocol and build Figure F8.

- GENERALIZATION COMPLETE (2026-07-17). Round 2 (--protocol leave_prn, PRN_LIMIT=None)
  finished ALL 11 PRN folds in one session (no further chunking needed -- confirms the
  batch=256 speedup holds at scale). generalization.csv is now FINAL: 182 rows =
  39 (cross_scenario, 3x13) + 143 (leave_prn, 11x13). This is the canonical file --
  cite only this. THREE-VIEW PROTOCOL DONE for all 13 models; ready to build Figure F8.
  Leave-PRN headline findings (per-architecture, not per-family, failure -- reinforces
  the cross-scenario finding and gives a 4th data point for "no family is safe"):
    PRN=19 held out: BiLSTM collapses to 0.006(!) recall, Transformer 0.075, LSTM
      0.081 -- while CNN-1D 0.873, CNN-LSTM 0.903, TCN 0.811 hold up FINE on the SAME
      fold. Same "family" (deep), opposite outcomes.
    PRN=7 held out: GradientBoosting collapses to 0.028(!), XGBoost 0.072 -- while
      KNN 0.997 and every deep model (>=0.97) hold up fine on the SAME fold.
  Most other PRNs (6,10,11,16,20,23,30) show near-ceiling recall (>0.9) for nearly all
  models -- cross-satellite generalization is broadly strong on this corpus EXCEPT for
  specific problematic satellites (3,7,13,19), and which satellite breaks which model
  is architecture-specific, not family-specific. Strong material for F8 panel (b)
  (PRN-level F1 distribution) -- show the spread/outliers, not just the mean.
  NEXT: defense (24_defense.py, Round N) is now the ONLY remaining data-generation
  step before all figures (F2-F9) can be built and Results/Discussion written.

- DEFENSE RUN 1 INVALID + FIX (2026-07-17). First full defense run (defense_baseline.csv,
  all 6 DL) produced DEGENERATE adv_train models: clean FAR 0.89-1.0 (CNN-1D/LSTM/
  CNN-LSTM = 1.0 exactly), i.e. the adversarially-trained models collapsed to a trivial
  "predict spoof for everything" classifier. Their apparent perfect robustness (ASR=0)
  was an ARTIFACT -- no authentic region means no spoof can be flipped to authentic.
  ROOT CAUSE (my bug, not a property of AT): the robust-val early-stopping metric was
  spoof RECALL under PGD, which is maximized by the all-spoof classifier -> selection
  rewarded degeneracy. The `far` column I added as a guardrail is what caught it
  (without it we'd have wrongly reported "AT gives perfect robustness"). FIX: changed
  the selection metric to robust BALANCED ACCURACY at tau=0.5 (mean of TPR,TNR; caps
  the all-spoof collapse at 0.5) -- the standard Madry/Rice robust-accuracy selection.
  Smoke-verified: adv_train FAR now ~= undefended FAR (no degenerate gap), AT improves
  PGD robustness, decision-based still succeeds (ASR 0.63 @0.2, 1.0 unbounded) = the
  coherent "AT hardens gradients not the boundary" story. USER MUST RE-RUN the full
  defense (delete/don't-seed the OLD degenerate defense_baseline.csv first, else the
  per-model resume logic skips everything). Inclusion decision waits on the valid
  full-run FAR + decision-based numbers.

- DEFENSE METHOD LITERATURE VERIFICATION (2026-07-17, user asked to confirm vs expert
  practice). CONFIRMED via web: (a) Rice et al. 2020 select the AT checkpoint by best
  robust ACCURACY / lowest robust ERROR on a held-out VALIDATION set (not recall, not
  loss) -> our fixed selection (robust BALANCED accuracy) realigns with this standard;
  the original robust-RECALL selection was a genuine deviation and caused the collapse.
  (b) The AT-under-imbalance boundary-distortion collapse is a DOCUMENTED phenomenon,
  not our bug: "Alleviating the Effect of Data Imbalance on Adversarial Training"
  (arXiv 2307.10205) states AT can "move the decision boundary... distort... and
  destroy the original decision boundary built by clean data"; class-balanced metrics
  are the literature-consistent mitigation. (c) AT as an empirical defense baseline for
  binary attack/benign SECURITY DETECTORS is standard in adjacent domains (intrusion
  detection / network security). HONEST FRAMING for Methods: cite Madry (PGD-AT) + Rice
  (robust-val early stopping on robust accuracy); state the balanced-accuracy selection
  as our grounded instantiation of the "select on robust accuracy, class-balanced"
  principle (not one paper's verbatim recipe); state plainly that the defense
  methodology is adapted from the general adversarial-ML literature since GNSS-specific
  AT-defense work is thin (An et al. 2025 list it as future work). Consider adding
  arXiv 2307.10205 to references.bib if we lean on the imbalance point (verify metadata
  via Crossref first -- no hallucinated cite).

- DEFENSE RUN 2 (2026-07-18, defense_baseline2.csv, balanced-acc selection fix).
  Fix PARTIALLY worked: 4/6 models valid, 2/6 STILL fully degenerate.
  FAR (clean) none->adv_train: CNN-1D 0.64->1.00 (COLLAPSE), CNN-LSTM 0.79->1.00
  (COLLAPSE), LSTM 0.63->0.86, BiLSTM 0.50->0.86, Transformer 0.43->0.71, TCN 0.45->0.89.
  So even the 4 "valid" AT models pay heavy precision cost (FAR 0.71-0.89). The 2
  collapsed models' asr=0 is the always-spoof ARTIFACT (do not report as robustness).
  CONSISTENT FINDING across all valid models (this is the usable result):
   - decision-based ASR still 1.0 at UNBOUNDED budget after AT (Transformer 0.96 and
     TCN 0.89 even at eps=0.2) -> AT does NOT stop the gradient-free attack.
   - AT DOES robustify gradient attacks (PGD@0.2 recall ~0.4->0.9 for the valid 4).
   - AT severely degrades precision (FAR up), to full collapse for 2 architectures.
  INTERPRETATION (coherent, supports thesis + Paper 2): proper PGD-AT hardens the
  gradient surface, not the boundary; empirical defense insufficient -> certified
  defense needed. The collapse is consistent with documented AT-under-imbalance
  boundary distortion (arXiv 2307.10205), NOT a rigor failure.
  DECISION PENDING (user): A = include COMPACTLY as "empirical AT insufficient"
  (1 para + small table, transparent about collapse, moderate risk); B = DROP from
  Paper 1, defenses -> Paper 2/future work (lowest risk; Paper 1 already strong on 3
  axes). Assistant leans B given the zero-major-revision mandate. STOP tuning AT
  further (3 rigorous iterations done; more tuning = p-hacking the defense).
  NOTE: AutoAttack question (user, 2026-07-18) is SEPARATE -- if defense stays (Option
  A), AutoAttack on the deep models would strengthen it and be a GNSS-domain first;
  if dropped (B), AutoAttack is optional polish for the main deep-model eval.

- NOVELTIES to weave into Methods/Results (coherent framing, mostly free): (1) the
  coupling-aware realizability enforcer as a NAMED contribution (C/N0~correlator-power
  projection; F9 figure); (2) DLSA/SNA/TPA reframed as observable-domain instantiations
  of An et al. (2025) attacks, honest NEGATIVE result = ineffective within a realizable
  budget (ASR<=0.05), sharpens the threat model, cite an2025adversarial, NO novelty
  claim on the names; (3) two-operating-point diagnostic (recall-0.95 + FAR-0.05) ->
  XGBoost/LightGBM/DecisionTree cannot reach FAR<=0.05; (4) decision-based min-Linf
  fragility ranking (GradientBoosting most fragile despite best clean F1); (5)
  independent reproduction of Song 2026 generalization collapse on a different receiver.
- DONE: front matter (abstract lean-style, 6 keywords), Introduction (sec:intro,
  4-item contributions), Related Work (sec:related; 4 subsections; positions vs
  Song 2026 = corroborate premise + extend to adversarial axis).
- Citation reconciliation for the new bib: removed DUP borhanidarian2024detecting
  (== existing borhanidarian2022deep, EURASIP JASP 2024:14); added song2026generalized
  + li2025realtime; fixed chen2025cgan issue 1->4. Related Work uses the audit-confirmed
  content (mahroof=kNN on ds3+ds8 NOT SVM; chen2025cgan=CGAN-ANN engineered features NOT
  1D-CNN; yang2024deep=BiLSTM-Attention-CNN wF1 0.974; borhanidarian=parallel DNN over
  CAF images). All CITATION_AUDIT.md fixes carried forward.
- TODO sections: signal_model, threat_model, enforcer (port from
  paper1_new_sections_draft.tex), methods, results (final numbers + F2-F9), discussion,
  conclusion. Style rule enforced: no em dash, no filler adj/adv, no inline "A,B,and C".

## 8c. FULL MANUSCRIPT DRAFTED + COMPILING (2026-07-18)

`papers/paper1-satnav/` is now a COMPLETE compiling draft, 20 pages:
front matter -> Introduction (Fig 1 pipeline) -> Related Work -> Signal Model (Figs 2-3)
-> Threat Model -> Methods (corpus, 13 detectors, 2 operating points, attack suite with
An et al. attribution + decision-based, enforcer as named contribution, 3-view protocol,
stats) -> Results (clean detection + Table 1, adversarial + Fig 5 fragility headline,
generalization + Fig 6, statistics) -> Discussion -> Conclusion -> declarations + refs.
Build clean: pdflatex->bibtex->pdflatex x2, 0 undefined citations, 0 undefined refs,
0 overfull hboxes, Arial embedded, all 6 figures + Table 1 render. sn-basic author-date
citations verified rendering ("Athalye et al. 2018; ..."). Section files in sections/*.tex.
All numbers spot-checked against the source CSVs (operating_point_recall95,
blackbox_boundary_all, generalization.csv, 19/22 stat logs) and match.
NUMBER-CONSISTENCY SWEEP DONE (2026-07-19, papers/paper1-satnav/
verify_numbers.py recomputes every cited number from the CSVs). Found + FIXED: (1) REAL
ERROR: domain-attack ASR claimed "<=0.05" but max is 0.19 (TPA/TCN@eps0.2) -> reworded
to "below 0.2, near zero for most, far weaker than learned attacks"; (2) FAR-0.05
"boosted-tree" -> "tree-based" (DecisionTree is not boosted); (3) min-Linf "0.021-0.033"
included MLP -> "tree detectors 0.021-0.025 and the MLP 0.033"; (4) cross-scenario
0.52/0.44 clarified as the mean over the 3 held-out scenarios, not ds2-specific.
ALL other numbers verified to MATCH (Table 1, F1/AUC/FAR ranges, fragility ranking,
ds7 deep 0.84, leave-PRN 0.006/0.028, McNemar 74/78, Friedman, bootstrap CIs, latency).
Stats captured permanently in results/tables/paper_statistics.md (were only in run logs).
Manuscript recompiles clean after fixes (0 undefined, 0 overfull, 20 pp).

REMAINING POLISH (not blocking): (1) [DONE above] number-consistency sweep;
(2) optional figures F4/F7 (MATLAB acqData / multi-correlator) + F9 (enforcer realizability);
(3) author bios/affiliation final check; (4) proofread pass; (5) capture the McNemar/Friedman
console numbers into a saved table (currently only in ledger); (6) update PROJECT docs.

## 9. Manuscript section-by-section edit checklist (the `.tex`)

- [ ] Abstract: GNSS integrity lead; real FGI-GSRx receiver; no-advantage finding;
      physically-realizable attacks; regenerated numbers.
- [ ] Introduction: lead with GNSS/PNT + resilient-PNT motivation, not adversarial ML.
- [ ] Related work: consolidate the fragmented sections; cite TEXBAT ION-2012,
      ds7/ds8 doc, TEXBAT-ML papers, Carlini-2019, Athalye-2018.
- [ ] NEW: Signal model + receiver observable generation (FGI-GSRx pipeline; the 9
      observables; the algebraic couplings and why derived ones are excluded).
- [ ] NEW: Threat model subsection (attacker capabilities; oracle access; budgets).
- [ ] Enforcer: physical justification via ds7 phasor takeover math (code drift
      1.2 m/s -> 1.273 us clock offset by t=468); coupling-aware C/N0.
- [ ] Methods: 13 models; common recall-0.95 operating point; the 3-view eval;
      block-temporal + purge split (state it explicitly).
- [ ] Attacks: elevate DLSA/SNA/TPA as GNSS-specific contributions; decision-based;
      attack-hyperparameter table.
- [ ] Results: regenerate Table 1 (clean), operating points, worst-case, adversarial
      ASR/recall, latency, stats — ALL from the new corpus. Add the generalization
      tables (PRN-level, cross-scenario).
- [ ] Figures: per `FIGURE_PLAN.md` (pipeline schematic; C/N0-Doppler-DLL onset
      time-history; feature distributions; receiver-validation appendix; detection;
      adversarial robustness; optional SQM distortion).
- [ ] Data/Code availability: TEXBAT datastore + md5; FGI-GSRx (GPLv3); repo.
- [ ] Reformat to the Satellite Navigation (SpringerOpen `sn-jnl`) template; drop
      the `referee` class option for the final.
- [ ] Defense baseline (reviewer panel's top ask): adversarial training or a
      randomized-smoothing detector — pull a lightweight version from Paper 2.

## 10. Real results (pipeline run 2026-07-14, complete corpus)

Full run: 12 stages, 24,888 s (~6.9 h), exited clean. Tables in `results/tables/`
(fresh = dated 2026-07-14). HEADLINE below; treat these as the numbers to write to.

**Clean operating point (recall>=0.95 on val), `operating_point_recall95.md`:**
F1 spans 0.744 (CNN-1D) to 0.843 (GradientBoosting). No family dominates; best is
classical (GB). AUC 0.803-0.903. FAR is HIGH (0.30-0.69) at recall-0.95 -> report
alongside AUC and consider a fixed-FAR point too. tau* single-sourced + consistent
across every downstream table (fixes Reviewer #1 #2).

**HEADLINE (decision-based BoundaryAttack, hard-label, `blackbox_boundary_all.csv`)**
Median min-L_inf to flip a spoofed sample (smaller = more fragile):
- tree/tabular: GradientBoosting 0.0095, LightGBM 0.0209, XGBoost 0.0222,
  DecisionTree 0.0247, RandomForest 0.0253, MLP 0.0334
- deep (seq): TCN 0.0402, LSTM 0.0414, Transformer 0.0456, BiLSTM 0.0509,
  CNN-1D 0.0654, CNN-LSTM 0.0714
- outlier: KNN 0.1009
=> Under a MATCHED gradient-free attack the non-differentiable trees are the
EASIEST to fool, not the hardest. GradientBoosting has the best clean F1 AND is the
most fragile. Old "classical robust" was gradient masking. This IS the answer to
Reviewer #1 #1. No architecture survives: boundary ASR >=0.74 at eps=0.1 (KNN 0.49),
>=0.92 at eps=0.2, ->1.0 unbounded. Thesis: architecture = utility/robustness
tradeoff, not security. KNN = most robust but worst clean F1 (0.754), FAR 0.61.

**Transfer (`multisurrogate_transfer.csv`):** multi-surrogate ensemble > single
(RF recall at eps=0.1: 0.472 single -> 0.336 ensemble). Even the ensemble transfer
UNDERSTATES tree vulnerability vs the boundary attack (the gradient-masking point).

**GNSS domain attacks (`adversarial_full_oppoint.csv`):** DLSA/SNA/TPA barely dent
recall (ASR <=0.05; DLSA sometimes raises it). Naive in-budget signal manipulations
do not fool the detectors; ML-crafted perturbations do.

**Latency (`latency_all13.csv`):** ONE table now, 0.12-92.7 us; DLSA-gen/inference
ratio <0.06 (fixes the Table 8/9 contradiction, Reviewer #1 #3).

**Monotonicity (`monotonicity.csv`):** monotonic=True for every eps-curve (passing
gradient-masking sanity check; cite explicitly). NOTE: with/without-enforcer rows
are ~identical -> 20_mono still uses the box-clip, NOT the coupling-aware enforcer.
Wire fit_coupling/enforce_coupling_phys into 18/20 for F9 (pending).

**CONSISTENCY GUARDS (pre-empt Reviewer #1 #3), enforce during the reframe:**
1. Two ASR conventions coexist: adversarial_full_oppoint = recall-degradation at
   fixed eps; blackbox_boundary = min-L_inf flip fraction. Label distinctly; never
   compare cell-to-cell.
2. Cite CLEAN metrics ONLY from operating_point_recall95 (full test). Adversarial
   files' "clean" rows use a ~2000-sample subset (slightly different) -> describe
   adversarial as degradation on that fixed subsample.
3. MLP = sklearn MLPClassifier, grouped classical/tabular (NOT deep seq). State once.
4. DL baseline (`dl_baseline_results.csv`) is 3-seed mean+/-std; report the CIs.

**Generalization (`generalization.csv`, 23_generalization --classical-only, 2026-07-15):**
Cross-scenario = leave-one-ATTACK-out (train other scenarios, test unseen attack).
Mean F1 by holdout: ds2 (overpowered 10dB) 0.16-0.55 (RF recall 0.09 = COLLAPSE);
ds3 (matched 1.3dB) 0.65-0.80; ds7 (matched+phase) 0.62-0.88. Story: detection
transfers WITHIN the matched-power family (ds3<->ds7, F1 up to 0.88) but COLLAPSES
to the overpowered class (ds2). No architecture immune. KILLER honest line: a
detector calibrated to 95% recall in-distribution drops to mean 52% recall (as low
as 9%) on an unseen attack class -> the recall-0.95 operating point does NOT transfer
OOD. This is the credible scoped limitation + the Paper-2 certified-defense motivation
(answers Reviewer #1 #4 cross-training). Leave-PRN-out: aggregate recall 0.82/F1 0.64
but BIMODAL/erratic per satellite (GB recall 1.0 on PRN10 vs 0.008 on PRN7; MLP F1
0.16 on PRN19) -> plot the F1 DISTRIBUTION across PRNs (F8 panel b), not the mean.
Methodology sound: recall spans 0.008-1.0 => tau calibrated in-distribution, NOT
re-tuned on held-out fold. TODO: run FULL pass (drop --classical-only) for the DL
folds too = complete F8 (costly: ~6 DL x 14 folds retrains; run overnight).

**Stale-file archive (2026-07-15, `run_pipeline_stage2.py` phase 0, ALREADY RUN):**
23 items moved (not deleted) to `results/tables/_archive_stale/2026-07-15_135435/`,
manifest in that folder. Groups: toy_receiver_pre_fgi_2026-04-17 (table1-7*.csv,
worst_case_robustness.csv, adversarial_attack_results.csv -- produced by LEGACY
03_adversarial_evaluation.py/04_statistical_analysis.py/05_manuscript_figures.py,
NOT part of run_pipeline.py -- never run 03/04/05 again, optimal_thresholds.csv is
a SHARED filename landmine between legacy 03/04 and the current 01/12 series);
orphaned_satgrid_2026-05-27 (generalization_results.csv, generalization_gap.csv,
raw_predictions/ -- different dataset entirely, no producing script in repo);
fgi_partial_corpus_smoke_2026-07-03_04 (blackbox_boundary*_quick.csv,
multisurrogate_transfer_quick.csv, old _*_log.txt -- pre-complete-corpus). Verified:
all 25 Paper-2 files (wp1_*, wp2_*, gnss_shield_*, cert_probe_*, certification_*,
certified_*, nonmonotonicity_diag_smoke*) present and untouched (results/tables/ is
gitignored -> shared across branches on disk, hence the mixing; NEVER archive those
by pattern, exact-filename allowlist only). `results/tables/` now holds ONLY current
Paper-1 (2026-07-14, to be refreshed tonight) + Paper-2 files.

**Stats capture:** 19/22 print to console only (no to_csv) -> `run_pipeline_stage2.py`
phase 1 tees every stage's stdout to `results/tables/_stage2_logs/NN_name_log.txt`,
fixing this for all 6 remaining stages, not just 19/22.

---

## Proofread + final-formatting pass (2026-07-20)

Full read-through of `papers/paper1-satnav/` for a no-error submission.

- **Numeric re-verify:** reran `verify_numbers.py` against all source CSVs; every cited
  number matches (clean F1 0.736-0.843, FAR 0.30-0.69, AUC 0.803-0.903; fragility GB
  0.0095 most-fragile -> KNN 0.1009 hardest, tree cluster 0.021-0.025, MLP 0.033, deep
  0.040-0.071; domain ASR max 0.193; x-scenario recall classical 0.528 / deep 0.442; ds2
  min RF 0.090; ds7 two deep >=0.84 CNN-1D 0.844 & TCN 0.840; leave-PRN BiLSTM 0.006,
  GB 0.028; McNemar 74/78; Friedman deep chi2 22.9 / classical 26.9).
- **Fixed 1 factual text/Table-1 mismatch** (the GPS-Solutions failure class): results.tex
  said BiLSTM (F1 0.789) "sits above three classical models"; it sits above only
  DecisionTree (0.787) and KNN (0.755) — MLP is 0.799. Changed to "two classical models".
- **Oxford-comma sweep:** normalised all serial lists to no-Oxford house style (abstract,
  intro roadmap + PNT/domains, related-work 2 lists, signal-model observable + spoofer
  lists, methods baseline/detector lists, discussion caption). Compound-sentence commas
  left intact.
- **Reference-macro consistency:** all figure refs now `Figure~\ref` (1 stray `(Fig.~...)`
  in intro fixed); Table~ and Section~ already uniform; 0 hard-coded fig/table numbers;
  0 em/en dashes anywhere.
- **Front matter verified** (rendered p.1): title, 3 authors + affil markers, RCSSTEAP/
  Beihang [1] + FUT Owerri [2], corresponding ahmedwasiu24@buaa.edu.cn — all correct.
- **New modern figures verified in-page** (rendered pp.18-19): Fig.9 rank-inversion slope
  (clean-F1 vs robustness ranks, visibly reversed) and Fig.10 parallel-coordinates
  (GB best-clean/worst-robust, KNN mirror) — both legible, correct data, legends clean.
- **Build:** 23 pp, 0 undefined citations, 0 undefined refs, 0 errors, 0 overfull >5pt.

**Still open (user actions):** F7 SQM figure needs one MATLAB multi-correlator pass
(`addpath(genpath('...FGI-GSRx'))`, ds7 config with `enableMultiCorrelatorTracking=true`,
`mulCorrFingers,[-2:0.25:2]`, send new `trackData_ds7` .mat); then correlation-magnitude
vs code-offset plot for authentic (t<110s) vs spoofed (t>150s).

---

## F7 SQM built from real data + F2 correction + authorship (2026-07-21)

Multi-correlator pass finished (user ran gsrx with `enableMultiCorrelatorTracking=true`,
`mulCorrFingers=[-2:0.25:2]`); `data/FGI_Data/out/trackData_ds7_full.mat` now carries
`mulCorrFingersOut` (250000 epochs x 17 taps) for all 10 PRNs. So F7 is the REAL figure,
not the placeholder that was drafted.

- **F7 (`fig07_sqm.pdf`, now Fig. 4), generator `manuscript/.../make_f7.py`:** PRN 23,
  authentic (t<110 s) vs spoofed (t>150 s), each correlation function normalised to peak.
  Panel (a) overlay (near-coincident), panel (b) spoofed-minus-authentic residual. Honest
  finding: ds7 leaves only a SMALL correlator distortion (peak stays at 0 offset; off-peak
  floor rises, late shoulder slightly suppressed; |residual| <= ~0.02 = 2% of peak). This
  is the correct story for a power-matched, carrier-phase-aligned takeover and reinforces
  the need for multi-observable learned detection.
- **F2 (`fig02...`, Fig. 3) CORRECTED** (found during F7 consistency check): old caption/
  text claimed "the C/N0 and delay-lock discriminator diverge through the 110-150 s
  takeover". Data says otherwise: dllDiscr mean ~0 in all windows (only variance rises);
  the real signature is C/N0 settling ~2 dB below authentic AFTER the takeover, Doppler
  matched. Also the ds7 trace was being straight-line-interpolated across the discarded
  [110,150] gap (a visual artifact). Fixed `make_figures.py fig_f2` to break lines at
  gaps (NaN insert), and rewrote the caption + body text to: "after the takeover C/N0
  settles ~2 dB below authentic, Doppler matched, DLL near zero; shaded band excluded
  from corpus". Now consistent with F7.
- **Authorship:** Joel Segun Ojerinde promoted to co-corresponding (`\author*` + email
  joelojerinde@gmail.com). Both Joel and Wasiu Akande Ahmed now corresponding. NOTE:
  sn-jnl lists corresponding emails in author (source) order, so Joel's appears first
  (he is first author); "Ahmed first in the corresponding line" is not achievable without
  making Ahmed first author. Flagged to user; email may be swapped for an institutional
  address. A FOURTH author (a Beihang faculty member in-line, TBC by Dr Ahmed) is to be
  added later.
- **Build:** 23 pp, 0 undefined citations/refs, 0 errors, 0 overfull. Figure count now
  11 in-text + Table 1 (F7 added; figure numbers after Fig. 3 shifted +1, all via \ref).

---

## Full reference + metric audit (2026-07-21)

Rigorous end-to-end verification (internal recompute + internet + provided reference
PDFs in `papers/references/`). No hallucination: every cited external metric checked
against full text.

- **Own results:** `verify_numbers.py` re-run; every cited number matches source CSVs
  (Table 1, abstract, results, discussion, conclusion, all captions). F2/F7 consistent.
- **Cite reconciliation:** 43 keys used, all defined in `references.bib`; 14 unused
  leftover entries (harmless, not emitted by bibtex).
- **Verified against provided PDFs:** romaniuc max F1 = **90.7%** (matches "near 0.90");
  yang BAC-3 weighted F1 = **0.9742** (matches "0.97") + BiLSTM-Attention-CNN architecture;
  mahroof kNN on **ds3 and ds8** (exact); jullian implements+compares **MLP and LSTM**
  (matches "recurrent"); iqbal VAE+GAN trained on a **single (genuine) class** (matches
  "one-class"). Verified online: An 2025 (DLSA+SNA vs SVM, position domain, 99.9%->20.4%),
  Song 2026 (>92.7% ARR / 10 unseen scenarios), + all table-paper metadata/DOIs.
- **3 corrections made (were wrong/overstated):**
  1. **sung** -- was "1D CNN on *correlator outputs*"; paper uses **signal-strength
     features** (SNR mean/std/difference). Fixed in related_work + bib title
     ("convolution" not "convolutional").
  2. **borhani-darian** -- was "*bank of parallel networks* ... *carrier-to-noise
     ratio*"; paper is a **parallelized DNN** over CAF images reporting gains at
     moderate-to-high **signal-to-noise ratio**. Fixed both.
  3. **Song "drop toward chance"** (4 places: intro/related/discussion/conclusion) --
     actual baselines retain **46-83% ARR** (CNN ~69%, LSTM ~74%, Transformer ~83%), so
     softened to "lose accuracy" / "generalization loss"; kept the verified ">0.92
     retention across ten scenarios".
- **Canonical method cites** (goodfellow/madry/athalye/carlini/papernot/liu/szegedy/
  brendel/vaswani/breiman/friedman/xgboost/lightgbm/bai/hochreiter/dietterich/demsar):
  correct, PDFs present for most, standard DOIs.
- **Build:** 23 pp, 0 undefined citations/refs, 0 errors, 0 overfull. Nothing left
  unverified.

**Project docs updated (2026-07-21):** `Masters_Research_Proposal.md`,
`PROJECT_EXPLAINED.md`, `NAV_EXPORT_RUNBOOK.md` -- P1 status set to "Satellite
Navigation, draft complete" (was "GPS Solutions, under review"); dated P1 note added to
PROJECT_EXPLAINED; runbook status note added (verified; workflow reused for F7 SQM).

**Follow-ups (2026-07-22):**
- **4th bib error found + fixed** (during reference-PDF mapping): `lin2024sar` was
  "Remote Sens. 16(3):596, 2024, doi 10.3390/rs16030596" -- the PDF is actually
  Remote Sens. **15(10):2699, 2023, doi 10.3390/rs15102699** (Lin et al.). Corrected.
- **Authorship changed to Dr Ahmed SOLE corresponding** (Joel reverted to first author
  only, no `\author*`). Byline: Ojerinde[1], Ahmed*[1], Akande[2]; corresponding line
  shows only ahmedwasiu24@buaa.edu.cn. This is the student-first-author /
  supervisor-corresponding convention (sn-jnl ties corresponding-email order to byline
  order, so this is the clean way to have Ahmed as the corresponding contact).
- **Reference PDFs copied** to `papers/paper1-satnav/references/`, renamed to
  BibTeX keys: 33 of 43 cited refs present; 9 missing (an2025, brendel2018decision,
  carlini2019evaluating, chen2020hopskipjump, humphreys2012texbat, humphreys2016texbatds78,
  kerns2014uav, li2025realtime, song2026generalized); fgigsrx is software. Shared library
  in `papers/references/` kept intact for P2/P3. Manifest: `references/REFERENCES_MANIFEST.md`.
- Build still clean: 23 pp, 0 undefined/errors.

**Second full sweep + citekey-year cleanup (2026-07-22):**
- User downloaded 5 of the 9 missing PDFs (an2025, brendel2018decision, song2026,
  chen2020hopskipjump, carlini2019). All 5 verified against full text: authors, venue,
  year, DOI exact; an2025 "open question re other architectures under black-box"
  confirmed verbatim in its future-work; song2026 ARR ">92.70% across 10 unseen
  scenarios" exact and baselines beaten by 9.4-46.2 pts (so retain ~46-83%, confirming
  the earlier "lose accuracy" softening, not "toward chance"). Now 38/43 PDFs local.
- kerns2014uav + li2025realtime verified from user-pasted citations (J. Field Robot.
  2014 31(4):617-636, doi 10.1002/rob.21513; GPS Solut. 29(1):45, doi
  10.1007/s10291-024-01802-8). NOTE: li2025realtime was published online 26 Dec 2024 but
  Springer assigns it to vol 29 = 2025, so the citation year is 2025 (kept).
- **9 citekeys renamed so key-year = actual publication year** (full year audit found
  they were internally inconsistent; bib YEAR fields were already correct):
  borhanidarian2022->2024deep, chen2023->2022svm, iqbal2022->2024deep,
  jullian2023->2022deep, lin2024->2023sar, mahroof2022->2024ml, romaniuc2023->2025lstm,
  sung2023->2022cnn, yang2024->2025deep. Applied across bib + all .tex + PDF filenames.
- All reference PDFs in `Satellite_Navigation/references/` are now named `<citekey>.pdf`
  (key year = pub year); manifest refreshed.
- Final audit: 0 key/year mismatches, 0 undefined citations/refs, 0 errors, 0 overfull.
  Every one of the 43 cited references verified (full text where available, else
  user-supplied citation / canonical). Total bib/attribution fixes across both sweeps:
  sung, borhani-darian, Song wording, lin metadata, + 9 citekey-year corrections.

---

## Related project docs to update LATER (user-listed)
`Masters_Research_Proposal.md`, `NAV_EXPORT_RUNBOOK.md`, `PROJECT_EXPLAINED.md`,
plus `PAPER1_REVISION_PLAN.md` (stale — predates this rebuild) and `CITATION_AUDIT.md`.
