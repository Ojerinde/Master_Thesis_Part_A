# Defense baseline investigation (`experiments/24_defense.py`) — 2026-07-29

**Status: deferred, NOT included in Paper 1.** `discussion.tex` keeps its original
"future work" framing (empirical AT / certified approaches, evaluating whether
either restores robustness, is ongoing work) — no manuscript changes needed for
this decision. The code below is real, tested progress, kept for whenever this
gets picked back up (Paper 2, or a future Paper 1 revision), not thrown away.

## Background

`24_defense.py` is Paper 1's own diagnostic: "does a standard defense fix this?"
without claiming a certificate (that's Paper 2 — GNSS-Shield, certified
randomized smoothing). Its original docstring assumed a *partial* outcome
("adversarial training recovers some robustness... at a small clean-accuracy
cost"). Separately, Paper 2's own early exploration (memory: 2026-06-13) already
found that naive empirical defenses collapse to an all-spoof predictor under
attack, which became part of why Paper 2 pivoted to a certified/physics-constrained
approach. This investigation independently hit the same failure mode in a
different script, and initially couldn't tell which situation it was in.

## The bug

On the first real Kaggle run (all 6 DL models, default settings), CNN-1D and LSTM
both showed robust-val balanced accuracy frozen at *exactly* 0.500 from epoch 1
(a same-label-always classifier's balanced accuracy), and the resulting
`adv_train` model collapsed to "always predict spoof": clean FAR → 1.0, PGD
recall → 1.0 at every epsilon, decision-based ASR → 0.0 at every epsilon (a
constant-output model can never be flipped to "genuine" by any perturbation).

## Hypotheses tested, in order

All local, CPU, `--smoke` scale (8,000-row subsample, 3 epochs, 3 PGD steps) —
never confirmed at real scale (144,900 rows, up to 30 epochs, 10 PGD steps, GPU).

1. **Random initialization** (warm-start `adv_train_model` from the already-trained
   clean/undefended model instead of `build_model()`'s fresh random init).
   Reasoning: the very first adversarial batch is generated against a network
   whose decision boundary is still arbitrary. **Result: no effect on the
   outcome** (FAR still 1.0), though it did change the *training trajectory*
   (bal-acc varied instead of sitting flat at 0.500) — informative, not a fix.
   Reverted.

2. **BatchNorm train/eval mode during the PGD inner loop.** Pang et al. (2021),
   ["Bag of Tricks for Adversarial Training"](https://arxiv.org/abs/2010.00467)
   (ICLR) — their empirically-validated baseline recipe uses eval-mode BN while
   crafting adversarial examples, train-mode only for the weight update; plain
   PGD-AT's common default (train-mode throughout) "may blur the distribution"
   since BN's running stats get computed against ten increasingly-perturbed
   versions of the same batch. **Result: zero measurable effect** — identical
   bal-acc trajectory to the unfixed baseline. Kept anyway (documented
   default-risk mitigation, doesn't hurt, no evidence against it), but not the
   answer alone.

3. **Epsilon exceeds the natural fragility margin.** Shaeiri, Nobahari & Rohban
   (2020), ["Towards Deep Learning Models Resistant to Large Perturbations"](https://arxiv.org/abs/2003.13370) —
   adversarial training "fails to train... given a large, but reasonable,
   perturbation magnitude" once epsilon exceeds the natural class margin.
   Verified applicable: `AUG_EPS=0.10` exceeds *every* undefended DL model's own
   median decision-boundary distance from the main fragility results
   (0.041–0.067; see `results/tables/blackbox_boundary_ci.csv`). Tested
   `AUG_EPS=0.05` (monkeypatched, not committed): **fixed CNN-1D** (FAR 0.971,
   non-trivial clean/PGD/decision numbers) but **did not fix LSTM** (identical
   total collapse). Real effect, not the whole story.

4. **TRADES instead of plain Madry-style AT loss.** Zhang et al. (2019),
   ["Theoretically Principled Trade-off between Robustness and Accuracy"](https://arxiv.org/abs/1901.08573)
   (ICML) — the standard, heavily-cited fix for the mechanism common to all of
   the above: plain Madry AT trains *only* on the adversarial view of each batch
   (`loss = BCE(model(x_adv), true_label)`), with nothing anchoring the model to
   clean-data behaviour, so "always predict spoof" can be a genuine minimum of
   that objective once epsilon isn't small relative to the margin. TRADES adds
   an explicit natural-accuracy term and replaces the adversarial objective with
   a KL-divergence term pulling the adversarial prediction toward the model's
   *own* clean prediction:
   `loss = BCE(model(x_natural), y) + beta * KL(model(x_adv) || model(x_natural))`,
   inner maximisation over the KL term (not the true-label loss). Exact formula
   sourced from the [authors' reference implementation](https://github.com/yaodongyu/TRADES/blob/master/trades.py),
   not a paraphrase. **Result: best of all four attempts.**
   - CNN-1D: clearly non-degenerate (FAR 0.986 vs undefended 0.954, clean
     R=0.989, PGD@0.1 R=0.227, decision-based ASR=0.131 — the boundary attack
     can find real adversarial examples again).
   - LSTM: still shows the strict degenerate signature (FAR=1.0, decision-based
     ASR=0.0 at every eps) — **but** PGD recall is no longer flat at 1.000
     across all epsilons like every prior attempt; it now varies
     (0.976→0.793→0.211 as eps rises 0.05→0.1→0.2), suggesting the underlying
     model may not be fully collapsed. Leading unverified theory: the
     *classification threshold* (`tau_for()`, searched to hit recall≥0.95 on
     validation) may itself be landing in a degenerate region if the
     validation-set probability distribution is too clustered — an
     evaluation-side artifact, not necessarily a training-side one. **Not
     verified.**

## Binary-classification adaptation of TRADES

TRADES is specified for multi-class softmax + `nn.KLDivLoss`; every model here is
binary with a single logit (`BCEWithLogitsLoss`). Used the exact equivalence
`softmax([0, z]) == [1-sigmoid(z), sigmoid(z)]` to embed each scalar logit as a
2-class logit vector `[0, z]`, so the KL term reuses PyTorch's numerically-stable
`log_softmax`/`KLDivLoss` machinery unchanged rather than hand-deriving and
separately stabilising a Bernoulli-KL formula. See `to_2class()` in
`adv_train_model()`.

## Current code state (committed)

`experiments/24_defense.py` implements TRADES (not plain PGD-AT):
- BatchNorm eval-mode during the inner loop (kept from hypothesis 2)
- Binary KL-divergence adaptation via `to_2class()`
- `TRADES_BETA = 1.0` (Zhang et al.'s own reference default; they explore [1, 10])
- `AUG_EPS` still at 0.10 — the eps=0.05 test (hypothesis 3) was a monkeypatched
  diagnostic, never committed to the file, since TRADES alone got further than
  eps-reduction alone. **Whether TRADES + reduced eps together resolves LSTM is
  untested.**

## What's not resolved

- LSTM (and likely BiLSTM, CNN-LSTM, which share LSTM cells — see
  `models/deep_learning/lstm.py`, `sequence length = 1` is deliberate, so
  classic RNN vanishing/exploding-gradient-over-time is NOT the mechanism,
  ruled out directly from the architecture code) may still be collapsed at the
  final-classification level even under TRADES.
- Never tested at real scale / real GPU — only tiny CPU smoke tests throughout.
- `TRADES_BETA=1.0` is untuned.
- The old `defense_baseline.csv` / `defense_baseline2.csv` (Jul 17–18, predating
  this investigation) were deleted during a results cleanup and are **not
  recoverable** (checked, not in Recycle Bin) — whether this exact collapse
  predates this investigation entirely is unknown.

## If resuming this later

1. Check the `tau_for()` threshold-degeneracy theory for LSTM directly (print
   the validation-set probability distribution for LSTM/adv_train specifically).
2. Test TRADES at real scale on Kaggle, `MODELS="CNN-1D"` alone first (strongest
   local evidence) before committing GPU time to all 6.
3. Test TRADES + reduced `AUG_EPS` (e.g. 0.05) together for LSTM specifically.
4. Full citation list: Zhang, Yu, Jiao, Xing, Ghaoui, Jordan (2019), TRADES,
   ICML, arXiv:1901.08573. Pang, Yang, Dong, Su, Zhu (2021), Bag of Tricks for
   Adversarial Training, ICLR, arXiv:2010.00467. Shaeiri, Nobahari, Rohban
   (2020), arXiv:2003.13370. Rice, Wong, Kolter (2020), Overfitting in
   Adversarially Robust Deep Learning (robust-val early stopping). Madry,
   Makelov, Schmidt, Tsipras, Vladu (2018), ICLR, arXiv:1706.06083 (original
   PGD-AT).
