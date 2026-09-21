"""
Windowed loader for the TEXBAT track corpus (Paper 1, deep-model track).
=======================================================================

`load_track_splits` in loader.py returns one epoch per row: every detector,
classical and deep, sees the same nine measurements. That keeps the information
identical across all fourteen models, so a difference between them is
attributable to the decision rule. It also means the recurrent and attention
models are evaluated on a degenerate sequence: an LSTM at sequence length one
carries no state, and a convolution over the nine-measurement vector slides
along an ordering that is a convention of the feature list, not a physical axis.

This module builds the other half of that comparison: genuine temporal windows,
so the sequence architectures are exercised on the structure they were designed
for. Everything else is held to the contract `load_track_splits` already
established, because the two tracks must remain comparable:

  * Windows are built only inside one contiguous run, a (scenario, prn, segment)
    group ordered by t_sec. `segment` is pre_onset or post_onset, and no group
    carries both labels (checked: 0 of 70), so a window can never straddle the
    spoofing onset and every window has one unambiguous label.
  * The block-temporal cut is the same: first `train_frac` of a run to train,
    next `val_frac` to val, remainder to test, dropping `purge` epochs at each
    boundary. Windows are built inside a block and never cross a block edge, so
    no epoch of a training window appears in a test window.
  * Windows do not overlap (stride == window). Overlapping windows would repeat
    the same epochs across samples and inflate the apparent sample size, which
    is the same autocorrelation problem the epoch-level split exists to avoid.
  * The scaler is fitted on training epochs only and applied per timestep, so
    the windowed models operate in the same min-max [0, 1] space as the
    epoch-level models and attack budgets stay comparable.

Returns X with shape (n_windows, window, 9).
"""
import numpy as np

from data.loader import load_texbat_track


def load_track_windows(window=20, scenarios=None, train_frac=0.70,
                       val_frac=0.10, purge=20, train_stride=None, verbose=False):
    """Block-temporal, leakage-free split of temporal windows.

    Args:
        window: epochs per window. At the 20 Hz export cadence, 20 epochs is 1 s.
        train_stride: step between TRAINING window starts. Defaults to `window`,
            i.e. no overlap. A smaller stride yields more training windows from
            the same epochs, which matters because non-overlapping windowing
            reduces the sample count by a factor of `window` relative to the
            epoch-level track and could otherwise be mistaken for the effect
            under study. Validation and test windows never overlap regardless,
            so no test epoch is ever counted twice.

    Returns:
        X_train, X_val, X_test : float64, shape (n, window, 9), UNSCALED.
        y_train, y_val, y_test : int labels (0=genuine, 1=spoof).
        feature_names          : the nine observable column names.
        scaler                 : MinMaxScaler fitted on the training epochs.
        groups_train/val/test  : (scenario, prn, segment) per window, for
                                 cluster-aware resampling later.
    """
    from sklearn.preprocessing import MinMaxScaler

    df, feats = load_texbat_track(verbose=verbose, validate=True,
                                  scenarios=scenarios)
    df = df.reset_index(drop=True)

    blocks = {"train": [], "val": [], "test": []}
    groups = {"train": [], "val": [], "test": []}

    for key, g in df.groupby(["scenario", "prn", "segment"], sort=False):
        idx = g.sort_values("t_sec").index.to_numpy()
        n = len(idx)
        n_tr = int(round(n * train_frac))
        n_va = int(round(n * val_frac))

        # identical cut points to load_track_splits, so the two tracks agree
        spans = {
            "train": idx[:max(0, n_tr - purge)],
            "val":   idx[n_tr + purge:max(n_tr + purge, n_tr + n_va - purge)],
            "test":  idx[n_tr + n_va + purge:],
        }
        stride_tr = train_stride or window
        for split, span in spans.items():
            step = stride_tr if split == "train" else window
            for start in range(0, len(span) - window + 1, step):
                blocks[split].append(span[start:start + window])
                groups[split].append(key)

    X_all = df[feats].values.astype(np.float64)
    y_all = df["label"].values.astype(int)

    out = {}
    for split in ("train", "val", "test"):
        if blocks[split]:
            wi = np.stack(blocks[split])           # (n_windows, window)
            Xw = X_all[wi]                         # (n_windows, window, 9)
            # every epoch in a window shares one label by construction; assert it
            yw = y_all[wi]
            if not (yw == yw[:, :1]).all():
                raise AssertionError(
                    "a window spans more than one label; the segment grouping "
                    "no longer guarantees label homogeneity")
            out[split] = (Xw, yw[:, 0])
        else:
            out[split] = (np.empty((0, window, len(feats))), np.empty(0, int))

    Xtr, ytr = out["train"]
    scaler = MinMaxScaler().fit(Xtr.reshape(-1, len(feats)))

    if verbose:
        for s in ("train", "val", "test"):
            Xs, ys = out[s]
            print("  %-5s windows=%6d  spoof frac=%.3f"
                  % (s, len(Xs), ys.mean() if len(ys) else float("nan")))

    return (out["train"][0], out["val"][0], out["test"][0],
            out["train"][1], out["val"][1], out["test"][1],
            feats, scaler,
            np.array(groups["train"], dtype=object),
            np.array(groups["val"], dtype=object),
            np.array(groups["test"], dtype=object))


def scale_windows(X, scaler):
    """Apply an epoch-level scaler to (n, window, features) without reshaping loss."""
    n, w, f = X.shape
    return scaler.transform(X.reshape(-1, f)).reshape(n, w, f)
