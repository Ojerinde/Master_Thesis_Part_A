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

## Status
- F1: DRAFTED as TikZ in manuscript/GPS_Solutions_sn/paper1_new_sections_draft.tex.
- F2, F3: ready to build the moment `texbat_track_combined.csv` exists.
- F4: needs `acqData` saved from a tracked `.mat`.
- F5, F6: after models are retrained + attacks re-run on the new corpus.
- F7: optional; needs a multi-correlator tracking pass.
- F8: after the generalization driver runs (cross-scenario + leave-PRN-out).
- F9: after adversarial eval with/without the enforcer.
