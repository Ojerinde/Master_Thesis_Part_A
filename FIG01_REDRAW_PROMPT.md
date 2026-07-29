# Redraw prompt — Figure 1 (methodological pipeline)

Paste the block below into your image/diagram generator. It replaces the current
`figures/fig01_pipeline.jpg`, which is too busy and has two errors: it says "7
Classical Models" (now eight) and it omits the signal-validity enforcer. This
version is deliberately lighter (a short title plus one compact line per stage),
uses a pure white background, and uses only the two colours the rest of the paper
uses, blue for classical and orange for deep, so Figure 1 matches Figures 5 to 12.
Every label below is the exact text that must appear, spelled exactly as written.
Export as vector PDF (or 400 dpi PNG) at 16 cm width and save as
`figures/fig01_pipeline`.

---

## PROMPT

Create one clean, flat, minimalist technical pipeline diagram on a pure white
background, as a single horizontal row of six stages read left to right. Keep it
uncluttered: each stage is a white rounded rectangle with a thin dark-grey outline
(#333333), a short bold dark-grey title, and at most one small line of lighter grey
text beneath it. Use only two accent colours, and only where noted: blue (#0072B2)
and orange (#E69F00), the same blue and orange used elsewhere in the paper for the
classical and deep detectors. Flat vector illustration only: no gradients, no
shadows, no 3-D, no photographic textures, no clip-art or stock icons, no
background. Clean sans-serif typography throughout (Arial or Helvetica). All text
must be real, correctly spelled English words, spelled exactly as specified.

Do not connect the stages with plain straight arrows. Instead, run one continuous
thin light-grey horizontal spine line through the vertical centre of the whole row,
passing behind every box, and place a single small solid dark-grey chevron (a right-
pointing triangle) on that spine in each gap between two stages, so the diagram
reads as one smooth left-to-right flow rather than six separate boxes.

Stage 1, title "Raw TEXBAT I/Q", one line beneath: "GPS L1 C/A, 25 Msps".

Stage 2, title "FGI-GSRx software receiver", one line beneath: "acquisition and
tracking".

Stage 3, title "Nine per-epoch observables", one line beneath: "C/N0, Doppler,
correlator I and Q, DLL, PLL and FLL".

Stage 4, title "Fourteen detectors", containing two small side-by-side rounded
chips: a blue-filled chip with white text reading "8 classical" and an orange-filled
chip with dark text reading "6 deep"; one line beneath the chips: "common operating
point, recall 0.95".

Stage 5, title "Shared attack suite", one line beneath: "FGSM, PGD, transfer,
DLSA, SNA, TPA, decision-based".

On the spine between Stage 5 and Stage 6, draw a small rounded-outline lozenge sitting
on the flow (a gate the chevron passes through), labelled "Signal-validity enforcer"
with one smaller line beneath it reading "physical realizability". Keep it clearly
smaller than the six stages, so it reads as a checkpoint on the flow, not a seventh
stage.

Stage 6, title "Evaluation", one line beneath: "three views, fragility, statistics".

Keep all six stage boxes the same height and aligned to one common baseline, equal
width, equal spacing, small uniform rounded corners, and generous white space so the
figure looks calm and clean. Proofread every acronym so it appears exactly as
written: FGSM, PGD, DLSA, SNA, TPA, C/N0, DLL, PLL, FLL, TCN, BiLSTM.

---

## "Does not look AI" and correctness check before accepting the image

- Pure white background; flat fills only; no gradients, glows, bevels, shadows.
- Only two accent colours, blue and orange, used only on the "8 classical" and
  "6 deep" chips; everything else white with dark-grey outline and text.
- One short title plus at most one line per stage; no long bulleted lists.
- The continuous spine with chevrons, not separate straight arrows.
- "8 classical" (not seven) present; the signal-validity enforcer lozenge present on
  the flow between the attack and evaluation stages.
- Every acronym spelled exactly as above; identical box height, corner radius,
  stroke width, and font on every stage.
