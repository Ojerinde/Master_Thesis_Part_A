"""
Two-track deep detector training, one command, one machine.
===========================================================

Track E (epoch)   : input (batch, 9)     -- the nine measurements of one epoch.
Track W (windowed): input (batch, W, 9)  -- W consecutive epochs.

Both tracks run through the SAME training loop, the same three seeds, the same
early stopping and the same representative-run selection. The only thing that
differs is the input representation and therefore the axis each architecture
traverses. That is the whole point: any difference between the tracks is
attributable to temporal context and not to training protocol or hardware.

Why retrain track E rather than reuse the stored checkpoints: the stored
artefacts under code/gnss_adversarial_research/results/models/deep_learning/
carry no record of the device or library versions that produced them. Training
both tracks here, on one machine, removes that as a confound and makes the pair
reproducible from a single documented environment.

    python experiments/02x_two_track.py --window 20

Outputs, per track, into results/models/two_track/:
    <model>_<track>.pt              representative weights
    <model>_<track>_scores.npz      val/test scores for the operating point
    summary_<track>_w<W>.csv        mean/std test F1 over seeds
    env.json                        versions and device, for the methods section
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.model_configs import get_config                        # noqa: E402
from data.loader import load_track_splits                          # noqa: E402
from data.window_loader import load_track_windows, scale_windows   # noqa: E402
from models.deep_learning_windowed import build as build_w         # noqa: E402

MODELS = ["cnn_1d", "lstm", "bilstm", "cnn_lstm", "transformer", "tcn"]
SEEDS = [42, 43, 44]


def build_epoch(name, n_feat, cfg):
    """The epoch-level networks, exactly as the published run used them."""
    from models.deep_learning.lstm import _LSTMNet
    from models.deep_learning.bilstm import _BiLSTMNet
    from models.deep_learning.cnn import _CNN1DNet
    from models.deep_learning.cnn_lstm import _CNNLSTMNet
    from models.deep_learning.tcn import _TCNNet
    from models.deep_learning.transformer import _TransformerNet
    if name == "lstm":
        return _LSTMNet(n_feat, cfg["lstm_layers"], cfg["dense_layers"], cfg["dropout_rate"])
    if name == "bilstm":
        return _BiLSTMNet(n_feat, cfg["bilstm_layers"], cfg["dense_layers"], cfg["dropout_rate"])
    if name == "cnn_1d":
        return _CNN1DNet(n_feat, cfg["conv_layers"], cfg["dense_layers"], cfg["dropout_rate"])
    if name == "cnn_lstm":
        return _CNNLSTMNet(n_feat, cfg["conv_layers"], cfg["lstm_units"],
                           cfg["dense_layers"], cfg["dropout_rate"])
    if name == "tcn":
        return _TCNNet(n_feat, cfg["num_filters"], cfg["kernel_size"],
                       cfg["num_blocks"], cfg["dense_layers"], cfg["dropout_rate"])
    if name == "transformer":
        return _TransformerNet(n_feat, cfg["num_heads"], cfg["ff_dim"],
                               cfg["num_transformer_blocks"], cfg["mlp_units"],
                               cfg["dropout_rate"])
    raise ValueError(name)


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_once(net, Xtr, ytr, Xva, yva, cfg, device):
    pos_weight = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())],
                              dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["learning_rate"])

    def mk(X, y, sh):
        return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32),
                                        torch.tensor(y, dtype=torch.float32)),
                          batch_size=cfg["batch_size"], shuffle=sh)
    tr, va = mk(Xtr, ytr, True), mk(Xva, yva, False)

    best, best_state, bad, ep = np.inf, None, 0, 0
    for ep in range(cfg["epochs"]):
        net.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(net(xb), yb).backward(); opt.step()
        net.eval(); tot = n = 0
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(device), yb.to(device)
                tot += crit(net(xb), yb).item() * len(yb); n += len(yb)
        v = tot / max(1, n)
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    net.load_state_dict(best_state)
    return net, best_state, best, ep + 1


def scores(net, X, device, batch=512):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch], dtype=torch.float32).to(device)
            out.append(torch.sigmoid(net(xb)).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def run_track(track, data, window, device, outdir):
    Xtr, Xva, Xte, ytr, yva, yte, n_feat = data
    rows = []
    for name in MODELS:
        cfg = get_config(name)
        print(f"\n[{track}] {name}", flush=True)
        runs = []
        for seed in SEEDS:
            set_seed(seed)
            net = (build_w(name, n_feat, window, cfg) if track == "windowed"
                   else build_epoch(name, n_feat, cfg)).to(device)
            t0 = time.time()
            net, state, vloss, eps = train_once(net, Xtr, ytr, Xva, yva, cfg, device)
            sv, st = scores(net, Xva, device), scores(net, Xte, device)
            f1 = float(f1_score(yte, (st >= 0.5).astype(int)))
            runs.append(dict(seed=seed, f1=f1, vloss=vloss, eps=eps, state=state,
                             sv=sv, st=st, secs=round(time.time() - t0, 1)))
            print("    seed %d  F1=%.4f  val_loss=%.4f  %d ep  %.1fs"
                  % (seed, f1, vloss, eps, runs[-1]["secs"]), flush=True)

        f1s = [r["f1"] for r in runs]
        rep = min(runs, key=lambda r: abs(r["f1"] - float(np.median(f1s))))
        tag = f"{name}_{track}"
        torch.save({"state_dict": rep["state"], "model": name, "track": track,
                    "window": window if track == "windowed" else 1,
                    "n_feat": n_feat, "config": cfg, "seed": rep["seed"],
                    "seeds_run": SEEDS}, outdir / f"{tag}.pt")
        np.savez(outdir / f"{tag}_scores.npz", val=rep["sv"], y_val=yva,
                 test=rep["st"], y_test=yte)
        rows.append(dict(model=name, track=track, n_seeds=len(runs),
                         test_f1_mean=float(np.mean(f1s)),
                         test_f1_std=float(np.std(f1s)),
                         representative_seed=rep["seed"], epochs_run=rep["eps"],
                         params=sum(p.numel() for p in net.parameters()),
                         seconds_total=round(sum(r["secs"] for r in runs), 1)))
        print("    mean F1 %.4f +/- %.4f  (rep seed %d)"
              % (rows[-1]["test_f1_mean"], rows[-1]["test_f1_std"], rep["seed"]),
              flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--tracks", default="epoch,windowed")
    ap.add_argument("--train-stride", type=int, default=None,
                    help="step between training window starts; defaults to --window (no overlap)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.outdir) if args.outdir else ROOT / "results" / "models" / "two_track"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  window={args.window}  tracks={args.tracks}")

    all_rows = []
    for track in args.tracks.split(","):
        if track == "epoch":
            Xtr, Xva, Xte, ytr, yva, yte, feats, sc = load_track_splits(verbose=True)
            data = (sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte),
                    ytr, yva, yte, len(feats))
        else:
            (Xtr, Xva, Xte, ytr, yva, yte, feats, sc, *_
             ) = load_track_windows(window=args.window,
                                    train_stride=args.train_stride,
                                    verbose=True)
            data = (scale_windows(Xtr, sc), scale_windows(Xva, sc),
                    scale_windows(Xte, sc), ytr, yva, yte, len(feats))
        rows = run_track(track, data, args.window, device, outdir)
        pd.DataFrame(rows).to_csv(
            outdir / f"summary_{track}_w{args.window}.csv", index=False)
        all_rows += rows

    pd.DataFrame(all_rows).to_csv(outdir / f"summary_both_w{args.window}.csv", index=False)
    (outdir / "env.json").write_text(json.dumps({
        "device": device, "python": sys.version.split()[0],
        "torch": torch.__version__, "numpy": np.__version__,
        "platform": platform.platform(), "seeds": SEEDS,
        "window": args.window, "train_stride": args.train_stride or args.window,
        "threads": torch.get_num_threads()}, indent=2))
    print("\nwrote", outdir)


if __name__ == "__main__":
    main()
