# Paper 1 — Manuscript Figure Plan (Satellite Navigation, IF 12.4)

Generation specs ("prompts") for each planned figure. Generate later, once the
corpus and the saved MATLAB outputs are in hand.

**Publication standard for all figures:** vector PDF (or >=300 dpi PNG), clean
sans-serif fonts, colorblind-safe palette, no chartjunk, and **no MATLAB-GUI or
console screenshots**. Data-driven figures are built with matplotlib from the
FGI-GSRx corpus `data/processed/texbat_track_combined.csv`.

Corpus columns available to figures:
`scenario, source_file, spoof_type, prn, epoch_idx, t_sec, segment, label,
label_name, cn0_dbhz, mean_cn0_dbhz, noise_cn0, doppler_hz, carr_freq_hz,
i_prompt, q_prompt, prompt_power, prompt_phase_rad, dll_discr, fll_filter,
pll_lock, fll_lock`

---

## MATLAB outputs to SAVE now (so the figures can be built later)

- **Keep every `FGI_Data\out\trackData_*_full.mat`** — do not delete. Each holds
  `acqData` (acquisition metrics) and `trackData` (per-epoch observable arrays).
- **`texbat_track_combined.csv`** — the exported per-epoch corpus (covers F2, F3).
- **F4 (receiver validation):** from a tracked `.mat`, save `acqData.gpsl1`
  (per-PRN acquisition metric) and, if possible, one PRN's acquisition
  correlation-vs-code-phase vector.
- **F7 (SQM, later):** run ONE tracking pass with
  `enableMultiCorrelatorTracking=true`, `multiCorrelatorTrackingChannel=[1:12]`,
  and save `trackData.gpsl1.channel(c).mulCorrFingersOut` (correlator taps).

---

## F1 — Pipeline schematic  [no data needed; can draft anytime]
**Purpose:** the paper's methodological backbone; shows the full signal-level chain.
**Spec:** left-to-right flow of clean boxes + arrows:
Raw TEXBAT IQ (25 Msps, GPS L1) -> FGI-GSRx software receiver (acquisition ->
carrier/code tracking loops) -> per-epoch observables (C/N0, Doppler, correlators,
PLL/FLL lock) -> ML detectors (7 classical + 6 deep) -> adversarial attacks
(FGSM, PGD, multi-surrogate transfer, DLSA/SNA/TPA, decision-based) ->
common recall-0.95 operating-point evaluation. Vector graphic.

## F2 — Spoofing-signature time-history  [MONEY FIGURE; corpus CSV]
**Purpose:** proves real signal-level work; shows the observable anomalies the
detector exploits (mirrors the TEXBAT paper's Fig. 9 diagnostic style).
**Spec:** 3 stacked panels vs `t_sec` (0-250 s) for ONE representative PRN present
in both ds7 and cleanStatic (e.g. PRN 23, high C/N0):
- (a) `cn0_dbhz`  - authentic (cleanStatic) vs spoofed (ds7) overlaid
- (b) `doppler_hz` - same overlay
- (c) `dll_discr`  - same overlay
Vertical dashed lines at the ds7 spoof onset (110 s) and full takeover (150 s),
with the 110-150 s transition band shaded. Two-colour legend (authentic/spoofed).
**Data:** filter corpus to `prn==23 & scenario in {cleanstatic, ds7}`.

## F3 — Feature distributions, genuine vs spoof  [corpus CSV]
**Purpose:** shows the discriminative signal in each observable.
**Spec:** small-multiples grid; one panel per feature
(`cn0_dbhz, doppler_hz, prompt_power, dll_discr, pll_lock, noise_cn0`), each a
split violin / KDE by `label` (genuine vs spoof), pooled over the static corpus
(cleanStatic + ds2 + ds3 + ds7). Colorblind-safe two-class palette.

## F4 — Receiver validation  [appendix; acqData]
**Purpose:** document the real software receiver locking onto authentic signals
(redrawn to manuscript style, NOT the raw MATLAB plot).
**Spec:** two clean panels: (a) acquisition metric per acquired PRN (bar chart);
(b) one PRN's acquisition correlation vs code phase, showing a single sharp peak.

## F5 — Detection performance  [eval results; after retraining]
**Purpose:** core clean-data result.
**Spec:** per-model precision / recall / F1 at the common recall-0.95 operating
point (grouped bars), 7 classical + 6 deep; optional ROC/PR overlay. From the
models retrained on the new corpus.

## F6 — Adversarial robustness  [eval results; after attacks]
**Purpose:** the central finding (no architectural robustness advantage).
**Spec:** worst-case recall (and/or decision-based ASR) vs epsilon in
{0.05, 0.10, 0.20}, deep vs classical families, matched attacks incl.
decision-based. From the adversarial eval at the operating point.

## F7 — Correlation-function distortion under spoofing  [optional; multi-corr pass]
**Purpose:** classic anti-spoofing signature (correlation-peak asymmetry/deformation).
**Spec:** correlation magnitude vs code-phase offset (multi-correlator taps) for one
PRN, an authentic epoch vs a spoofed epoch, showing the deformation. Requires the
multi-correlator tracking pass (see SAVE list).

## F8 — Generalization: cross-scenario + PRN-level  [eval results; added per audit #10]
**Purpose:** the reviewer-expected robustness evidence (TEXBAT-standard 3-view eval).
**Spec:** two panels. (a) Cross-scenario leave-one-out: detection F1 and worst-case
adversarial recall when the held-out scenario (ds2 / ds3 / ds7) is never seen in
training, grouped by model family. (b) PRN-level leave-PRN-out: F1 distribution over
held-out PRNs. Shows generalization to unseen attack types and satellites, not just
the mixed split. Data: the generalization driver (`load_track_splits(scenarios=)`
plus a leave-PRN-out mode).

## F9 — Adversarial realizability (enforcer effect)  [eval; added, supports Reviewer #4]
**Purpose:** show adversarial examples stay on the physically-valid manifold — the
"physical relation between signal control and feature perturbation."
**Spec:** 2D density of C/N0 vs I/Q power (10 log10(i^2+q^2)) for authentic, spoofed,
UNCONSTRAINED adversarial, and ENFORCER-CONSTRAINED adversarial examples. Unconstrained
points leave the physical C/N0~power band; enforced points stay on it. One figure that
directly answers the realizability objection.

---

## Status (updated 2026-07-18)
GENERATED to journal spec by `manuscript/Satellite_Navigation/make_figures.py`
(vector PDF + 400 dpi PNG; Arial embedded TrueType/fonttype-42; Okabe-Ito
colorblind-safe palette validated with the dataviz checker; sized to the 16 cm
Springer text width; every rendered figure visually inspected and defects fixed):
- F2 -> figures/fig02_signature_timehistory_prn23.pdf  (authentic cleanStatic vs
  spoofed ds7, PRN 23; C/N0 + Doppler + DLL; takeover band 110-150 s). DONE.
- F3 -> figures/fig03_feature_distributions.pdf  (genuine vs spoof, 6 observables). DONE.
- F5 -> figures/fig05_detection_operating_point.pdf  (F1 per model, sorted, by family;
  no-family-dominance story, F1 0.736-0.843). DONE.
- F6 -> figures/fig06_fragility_ranking.pdf  (HEADLINE: median min-Linf to evade,
  decision-based; GradientBoosting 0.009 most fragile, KNN 0.101 outlier; trees<deep). DONE.
- F8 -> figures/fig08_generalization.pdf  (a: cross-scenario recall heatmap 13x3,
  ds2 collapse column; b: leave-PRN strip, per-architecture outliers e.g. BiLSTM~0.006). DONE.
- F9 -> figures/fig09_realizability.pdf  (C/N0 vs log10(I^2+Q^2) manifold; unconstrained
  adversarial leaves the physical band, enforcer projects it back). DONE 2026-07-19,
  wired into Methods sec:enforcer (fig:realizability).
- F4 -> figures/fig04_receiver_validation.pdf  (a: acquisition metric per PRN, 10 acquired
  above threshold=10, PRN 23 strongest 44.7; b: PRN 23 code-phase correlation, single sharp
  peak). DONE 2026-07-19 from EXISTING FGI_Data/out/trackData_cleanStatic_full.mat via
  scipy.io (NO MATLAB re-run needed). Wired into Signal Model (fig:acq).
Journal-compliance confirmed: Arial present at C:/Windows/Fonts/arial.ttf (used).
- Old TikZ F1 draft superseded by the raster pipeline schematic (fig01_pipeline.jpg).

## Status (superseded — see the final paper)
This plan is complete and historical. All 11 figures are generated and wired into
the manuscript (`papers/paper1-satnav/figures/`, built by
`papers/paper1-satnav/make_figures.py`, the successor to this file's generation
script): F1 pipeline, F2 signature time-history, F3 feature distributions, F4
receiver validation, F5 detection operating point, F6 fragility ranking (with 95%
bootstrap CIs), F7 SQM correlation distortion (the multi-correlator pass below was
completed), F8 generalization, F9 realizability, plus two added during the peer-review
response: F10 rank inversion (clean vs adversarial ranking) and F11 the four-axis
trade-off. Current build: 25 pages, 0 undefined references, 0 overfull hboxes. For
the authoritative current state, read the manuscript
(`papers/paper1-satnav/`) and this repo's `PAPER1_MANUSCRIPT_UPDATES.md`
rather than this plan.

## Old status
- F1: DRAFTED as TikZ in manuscript/GPS_Solutions_sn/paper1_new_sections_draft.tex.
- F2, F3: ready to build the moment `texbat_track_combined.csv` exists.
- F4: needs `acqData` saved from a tracked `.mat`.
- F5, F6: after models are retrained + attacks re-run on the new corpus.
- F7: optional; needs a multi-correlator tracking pass.
- F8: after the generalization driver runs (cross-scenario + leave-PRN-out).
- F9: after adversarial eval with/without the enforcer.
