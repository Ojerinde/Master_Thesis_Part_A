# Redraw brief — Figure 1 (methodological pipeline)

This is a build specification for the paper's pipeline diagram
(`figures/fig01_pipeline.jpg`), written to be executed in a vector tool
(draw.io / diagrams.net, Inkscape, Adobe Illustrator, or PowerPoint exported to
PDF). **Do not use a text-to-image AI generator** — those garble small labels and
produce the exact "AI look" to avoid. If you prefer, this same spec can be
implemented directly in TikZ for a fully native vector figure; ask and it will be
provided.

## What is wrong with the current version (must fix)

1. **"7 Classical Models" → "8 Classical Models".** An RBF-kernel support-vector
   machine was added; the roster is now eight classical + six deep = fourteen
   detectors. The caption already says "eight classical and six deep"; the diagram
   must match.
2. **The signal-validity enforcer is missing.** It is a core contribution and must
   appear: every adversarial example is projected onto the physically realizable
   set before it reaches the detector. Show it as a distinct gate between the
   "Adversarial attacks" stage and the "Evaluation" stage.
3. **Replace the clip-art glyphs** (the little decision-tree and neural-net icons).
   They read as template/AI art and clash with the paper's flat figure style. Use
   either no icons or a single consistent set of thin line-icons.

## Content — six stages, left to right, one arrow between each

1. **Raw TEXBAT I/Q**
   - "Raw TEXBAT I/Q recordings"
   - sub: "25 Msps complex, GPS L1 C/A"
   - sub: "clean static + ds2, ds3, ds7 scenarios"

2. **FGI-GSRx software-defined receiver**
   - "FGI-GSRx software receiver"
   - two stacked sub-boxes with a down-arrow between them:
     "Acquisition" → "Carrier and code tracking loops"

3. **Per-epoch observables (nine)**
   - title: "Nine per-epoch observables"
   - list: "C/N0 (instant, mean, noise floor)", "Doppler", "Prompt correlator I, Q",
     "DLL discriminator", "PLL and FLL lock"

4. **ML detectors (fourteen)**
   - title: "Fourteen detectors at a common operating point (recall = 0.95)"
   - sub-box: "8 classical: random forest, gradient boosting, XGBoost, LightGBM,
     k-NN, MLP, decision tree, RBF-SVM"
   - sub-box: "6 deep: CNN-1D, LSTM, BiLSTM, CNN-LSTM, Transformer, TCN"

5. **Adversarial attacks** (this is the one accented stage — see palette)
   - title: "Shared attack suite"
   - list: "White-box FGSM, PGD", "Multi-surrogate transfer",
     "GNSS domain: DLSA, SNA, TPA", "Gradient-free decision-based (boundary)"

6. **Signal-validity enforcer** (new, small, sits as a gate between 5 and 6, or as a
   labelled band the attack arrow passes through)
   - title: "Physical-realizability enforcer"
   - sub: "projects every attack onto the C/N0–correlator-power manifold"
   - render it visually distinct (e.g. a thin outlined lozenge on the arrow) so it
     reads as a constraint every attack must pass through, not another stage.

7. **Evaluation**
   - title: "Evaluation protocol"
   - list: "Common operating point (recall 0.95)",
     "Three views: block-temporal, cross-scenario, leave-satellite",
     "Fragility = median min L-infinity to evade",
     "Bootstrap CIs, McNemar, Friedman"

## Style — match the paper's flat, colour-blind-safe house style

- **Palette (Okabe-Ito, same as figs 2–12).** Neutral stages: light grey fill
  `#EDEDED` (or white) with a dark-grey stroke `#5c5c5c`, ~1.2 pt. Text ink
  `#1a1a1a`. Use ONE accent, deep blue `#0072B2`, for the "Adversarial attacks"
  stage outline and the enforcer, since the attack suite and its realizability
  constraint are the paper's contribution. Avoid the current maroon/navy mix and
  avoid more than one accent colour.
- **Typography.** Sans-serif throughout (Arial / Helvetica). Stage titles bold,
  ~9–10 pt at final size; body ~7.5–8 pt. Never below 7 pt at the printed 16 cm
  (full text-width) size. One type family, two weights only.
- **Boxes.** Rounded rectangles, small uniform corner radius (~4 pt), consistent
  size, generous internal padding, no drop shadows, no gradients, no 3-D.
- **Arrows.** Thin (~1.2 pt) solid dark-grey arrows, single arrowhead, horizontal,
  equal length between stages.
- **Layout.** One clean left-to-right row (the current layout is fine); full
  journal text width (16 cm), height whatever keeps text ≥7 pt. Align all stage
  boxes to a common baseline and equal height.
- **Export.** Vector PDF with fonts embedded (or 400 dpi PNG as a fallback), sized
  to 16 cm width, to sit beside the matplotlib figures without a visible style
  break.

## "Does not look AI" checklist

- No clip-art or stock icons; if any icon is used, one consistent thin-line set only.
- No gradients, glows, bevels, or drop shadows — flat fills only.
- One accent colour, not a rainbow.
- Identical corner radius, stroke width, and font on every box.
- Real, spelled-out, correct labels (proofread every acronym: DLSA, SNA, TPA, PGD,
  FGSM, C/N0, DLL, PLL, FLL) — no invented or truncated text.
- Equal spacing and alignment; nothing "almost" aligned.
