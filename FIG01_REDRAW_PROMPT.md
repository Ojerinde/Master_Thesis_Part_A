# Redraw prompt — Figure 1 (methodological pipeline)

Paste the block below into your image/diagram generator. It replaces the current
`figures/fig01_pipeline.jpg`, which has two errors to fix: it says "7 Classical
Models" (now eight), and it omits the signal-validity enforcer (a core
contribution). Every label below is the exact text that must appear, spelled
exactly as written. Export the result as a vector PDF (or 400 dpi PNG) sized to a
16 cm text width and save it as `figures/fig01_pipeline`.

---

## PROMPT

Create one clean, flat-design technical pipeline diagram in a minimalist academic
style, on a pure white background, arranged as a single row of seven stages read
left to right, with a thin dark-grey arrow pointing right between each stage.
Use only three colours: dark navy (#0E2E5D) for all titles and outlines, one
accent steel-blue (#0072B2) used only for the two stages that are this study's
contribution (the attack stage and the enforcer), and a light grey (#EDEDED) fill
for every neutral box. Flat vector illustration only: no gradients, no shadows, no
3-D, no photographic textures, no clip-art or stock icons. Clean sans-serif
typography throughout (Arial or Helvetica), bold stage titles and lighter body
text. All text must be real, correctly spelled English words, spelled exactly as
specified below.

Stage 1, a tall rounded rectangle titled "Raw TEXBAT I/Q recordings", with two
lines of smaller text beneath: "25 Msps complex, GPS L1 C/A" and "clean-static
plus ds2, ds3, ds7 scenarios".

Stage 2, a rounded rectangle titled "FGI-GSRx software receiver", containing two
small stacked sub-boxes with a short downward arrow from the first to the second:
the top sub-box reads "Acquisition" and the bottom sub-box reads "Carrier and code
tracking loops".

Stage 3, a rounded rectangle titled "Nine per-epoch observables", with five short
lines listed beneath the title: "C/N0 (instant, mean, noise floor)", "Doppler",
"Prompt correlator I and Q", "DLL discriminator", and "PLL and FLL lock".

Stage 4, a rounded rectangle titled "Fourteen detectors at a common operating
point (recall = 0.95)", containing two sub-boxes stacked vertically. The top
sub-box is headed "8 classical" and lists "random forest, gradient boosting,
XGBoost, LightGBM, k-NN, MLP, decision tree, RBF-SVM". The bottom sub-box is headed
"6 deep" and lists "CNN-1D, LSTM, BiLSTM, CNN-LSTM, Transformer, TCN". (Note: this
must read eight classical detectors, not seven, and must include RBF-SVM.)

Stage 5, a rounded rectangle drawn with the steel-blue accent outline, titled
"Shared attack suite", listing four short lines: "White-box FGSM and PGD",
"Multi-surrogate transfer", "GNSS domain: DLSA, SNA, TPA", and "Gradient-free
decision-based (boundary)".

Between Stage 5 and Stage 6, on the connecting arrow, place a small steel-blue
outlined lozenge (a gate the arrow passes through) labelled "Signal-validity
enforcer", with one line of smaller text beneath it reading "projects every attack
onto the C/N0 and correlator-power manifold". Draw it clearly as a constraint every
attack must pass through, not as another full-height stage.

Stage 6, the final tall rounded rectangle titled "Evaluation protocol", listing
four short lines: "Common operating point (recall 0.95)", "Three views:
block-temporal, cross-scenario, leave-satellite", "Fragility = median minimum
L-infinity to evade", and "Bootstrap CIs, McNemar, Friedman".

Keep every box the same height and aligned to one common baseline, with equal
spacing and equal-length arrows between stages, small uniform rounded corners, and
generous internal padding. Do not add any decorative icons, logos, people, or
background. Proofread every acronym so it appears exactly as written: FGSM, PGD,
DLSA, SNA, TPA, C/N0, DLL, PLL, FLL, RBF-SVM, TCN, BiLSTM.

---

## Quick "does not look AI" check before you accept the image

- Every acronym spelled exactly as above; no invented or garbled words.
- Flat fills only; no gradients, glows, bevels, or drop shadows.
- Exactly three colours; the accent used only on the attack stage and the enforcer.
- Identical box height, corner radius, stroke width, and font on every stage.
- "8 classical" and "RBF-SVM" both present; the enforcer lozenge present on the
  arrow between the attack and evaluation stages.
- Equal spacing and clean alignment; nothing "almost" aligned.
