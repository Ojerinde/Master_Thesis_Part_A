"""
Train the six deep detectors on temporal windows (Paper 1, second track).
=========================================================================

The epoch-level track (02_deep_learning_baseline.py) gives every detector the
nine measurements of one epoch, so all fourteen hold identical information and a
difference between them is attributable to the decision rule. That design leaves
the sequence architectures unexercised: the recurrent models run for one step and
the convolutional and attention models traverse the feature ordering.

This script trains the same six architectures, with the same hyperparameters,
on windows of consecutive epochs, so the axis they traverse is time. Comparing
the two tracks answers whether temporal context changes what a transmitted
attack can do to a learned detector.

Runs on Kaggle GPU or locally. Writes one .pt per model plus a summary CSV.

    python experiments/02w_deep_windowed.py --window 20
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.model_configs import get_config                      # noqa: E402
from data.window_loader import load_track_windows, scale_windows  # noqa: E402
from models.deep_learning_windowed import build                  # noqa: E402

MODELS = ["cnn_1d", "lstm", "bilstm", "cnn_lstm", "transformer", "tcn"]

# The epoch-level track (02_deep_learning_baseline.py) trains each architecture
# under three seeds, reports mean and standard deviation, and keeps the run whose
# test F1 is closest to the median as the representative model. This track
# mirrors that protocol exactly; a difference in seed handling between the two
# would confound the comparison they exist to make.
N_SEEDS = 3
SEEDS = list(range(42, 42 + N_SEEDS))


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loaders(Xtr, ytr, Xva, yva, batch):
    def mk(X, y, shuffle):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                           torch.tensor(y, dtype=torch.float32))
        return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False)
    return mk(Xtr, ytr, True), mk(Xva, yva, False)


def train_one(name, Xtr, ytr, Xva, yva, window, n_feat, device, seed):
    cfg = get_config(name)
    set_seed(seed)
    net = build(name, n_feat, window, cfg).to(device)

    # class imbalance handled the same way the classical track handles it
    pos_weight = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())],
                              dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["learning_rate"])

    tr, va = loaders(Xtr, ytr, Xva, yva, cfg["batch_size"])
    best, best_state, bad = np.inf, None, 0
    t0 = time.time()

    for epoch in range(cfg["epochs"]):
        net.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(net(xb), yb)
            loss.backward()
            opt.step()

        net.eval()
        tot = n = 0
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(device), yb.to(device)
                tot += crit(net(xb), yb).item() * len(yb)
                n += len(yb)
        vloss = tot / max(1, n)

        if vloss < best - 1e-5:
            best, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break

    net.load_state_dict(best_state)
    return net, best_state, {"model": name, "seed": seed, "val_loss": best,
                             "epochs_run": epoch + 1,
                             "seconds": round(time.time() - t0, 1),
                             "params": sum(p.numel() for p in net.parameters())}


def scores(net, X, device, batch=512):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch], dtype=torch.float32).to(device)
            out.append(torch.sigmoid(net(xb)).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}   window: {args.window}")

    (Xtr, Xva, Xte, ytr, yva, yte, feats, scaler,
     gtr, gva, gte) = load_track_windows(window=args.window, verbose=True)

    Xtr, Xva, Xte = (scale_windows(a, scaler) for a in (Xtr, Xva, Xte))
    n_feat = len(feats)
    print(f"features: {n_feat}   train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")

    outdir = Path(args.outdir) if args.outdir else ROOT / "results" / "models" / "deep_windowed"
    outdir.mkdir(parents=True, exist_ok=True)

    from sklearn.metrics import f1_score

    rows = []
    for name in MODELS:
        print(f"\n--- {name} ---", flush=True)
        runs = []
        for seed in SEEDS:
            net, state, info = train_one(name, Xtr, ytr, Xva, yva, args.window,
                                         n_feat, device, seed)
            s_val, s_test = scores(net, Xva, device), scores(net, Xte, device)
            info["test_f1"] = float(f1_score(yte, (s_test >= 0.5).astype(int)))
            runs.append((info, state, s_val, s_test))
            print("    seed %d  F1=%.4f  val_loss=%.4f  %5.1fs"
                  % (seed, info["test_f1"], info["val_loss"], info["seconds"]),
                  flush=True)

        # representative run = the one closest to the median test F1, matching
        # the epoch-level track's selection rule
        f1s = [r[0]["test_f1"] for r in runs]
        med = float(np.median(f1s))
        info, state, s_val, s_test = min(runs, key=lambda r: abs(r[0]["test_f1"] - med))

        torch.save({"state_dict": state, "window": args.window, "n_feat": n_feat,
                    "config": get_config(name), "model": name,
                    "seed": info["seed"], "seeds_run": SEEDS},
                   outdir / f"{name}_w{args.window}.pt")
        np.savez(outdir / f"{name}_w{args.window}_scores.npz",
                 val=s_val, y_val=yva, test=s_test, y_test=yte)

        agg = {"model": name, "n_seeds": len(runs),
               "test_f1_mean": float(np.mean(f1s)), "test_f1_std": float(np.std(f1s)),
               "representative_seed": info["seed"],
               "epochs_run": info["epochs_run"], "params": info["params"],
               "seconds_total": round(sum(r[0]["seconds"] for r in runs), 1)}
        print("    mean F1 %.4f +/- %.4f   representative seed %d"
              % (agg["test_f1_mean"], agg["test_f1_std"], agg["representative_seed"]),
              flush=True)
        rows.append(agg)

    pd.DataFrame(rows).to_csv(outdir / f"train_summary_w{args.window}.csv", index=False)
    (outdir / f"meta_w{args.window}.json").write_text(json.dumps(
        {"window": args.window, "features": feats, "seeds": SEEDS,
         "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte),
         "scaler_min": scaler.data_min_.tolist(),
         "scaler_max": scaler.data_max_.tolist()}, indent=2))
    print("\nwrote", outdir)


if __name__ == "__main__":
    main()
