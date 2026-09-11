"""
Rigor check: is the S3 headline driven by the unexplained PRN 13 anomaly?

PRN 13 behaves anomalously under carrier-phase calibration (+2 dB where every other
channel sits between -0.06 and -0.72 dB), and three candidate explanations were
falsified: its code sidelobe at the noise finger is the minimum value, its samples
are verified coherently combined, and the spoofer-off control reproduces it exactly.
Every library point used phase calibration, so PRN 13 contributes roughly a tenth of
the rows behind every detection rate.

This re-scores the detectors over the capture-valid subset with and without PRN 13.
If the ranking and the evasion counts survive its removal, the result does not rest
on one misbehaving satellite and the anomaly can be reported as a bounded curiosity.

Scores only the 42 capture-valid points (Doppler-aligned, power advantage, actively
dragging the code), which is the authoritative subset, so it runs in a few minutes.

Output: results/tables/s3_prn13_check.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.metrics import recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.loader import load_track_splits
from config.model_configs import get_config
from models.deep_learning import (
    CNN1DModel, LSTMModel, BiLSTMModel,
    CNNLSTMModel, TransformerModel, TCNModel,
)

MODEL_ROOT = Path("D:/BEIHANG UNIVERSITY/Research/code/gnss_adversarial_research/"
                  "results/models")
LIB = Path("D:/BEIHANG UNIVERSITY/Research/data/FGI_Data/composite/"
           "realizable_library.csv")
OUT = Path(__file__).resolve().parents[1] / "results" / "tables"

FEATS = ['cn0_dbhz', 'mean_cn0_dbhz', 'noise_cn0', 'doppler_hz',
         'i_prompt', 'q_prompt', 'dll_discr', 'pll_lock', 'fll_lock']
TARGET_RECALL, EVADE_BELOW, SUSPECT_PRN = 0.95, 0.50, 13

SELECTED_CLASSICAL = {
    'RandomForest': 'RandomForest_default', 'XGBoost': 'XGBoost_default',
    'LightGBM': 'LightGBM_default', 'GradientBoosting': 'GradientBoosting',
    'KNN': 'KNN', 'MLP': 'MLP', 'DecisionTree': 'DecisionTree', 'SVM': 'SVM',
}
DL_REGISTRY = {
    'CNN-1D': (CNN1DModel, 'cnn_1d'), 'LSTM': (LSTMModel, 'lstm'),
    'BiLSTM': (BiLSTMModel, 'bilstm'), 'CNN-LSTM': (CNNLSTMModel, 'cnn_lstm'),
    'Transformer': (TransformerModel, 'transformer'), 'TCN': (TCNModel, 'tcn'),
}


def proba(model, X):
    p = np.asarray(model.predict_proba(X))
    if p.ndim == 2:
        p = p[:, 1]
    return p.ravel().astype(np.float64)


def threshold_for_recall(y, p, target=TARGET_RECALL):
    best = None
    for tau in np.unique(p)[::-1]:
        if recall_score(y, (p >= tau).astype(int), zero_division=0) >= target:
            if best is None or tau > best:
                best = tau
    return float(best) if best is not None else 0.0


def main():
    (_Xtr, Xv, _Xte, _ytr, yv, _yte, feats, scaler) = load_track_splits(verbose=False)
    Xv = Xv.astype(np.float64)
    Xv_s = scaler.transform(Xv).astype(np.float32)

    lib = pd.read_csv(LIB)
    # Authoritative subset: the published capture recipe requires the counterfeit to
    # be Doppler aligned (no beat), above authentic power, and actively dragging.
    pts = (lib.groupby('point')
              .agg(alpha=('alpha_db', 'first'), pull=('pulloff_mps', 'first'),
                   align=('align_frac', 'first')).reset_index())
    cap = set(pts[(pts['align'] == 1.0) & (pts['alpha'] > 0) & (pts['pull'] > 0)].point)
    sub = lib[lib.point.isin(cap)].reset_index(drop=True)
    print(f"capture-valid points: {len(cap)}; rows {len(sub):,}; "
          f"PRN13 rows {(sub.prn == SUSPECT_PRN).sum():,} "
          f"({100*(sub.prn == SUSPECT_PRN).mean():.1f}%)")

    X = sub[FEATS].values.astype(np.float64)
    Xs = scaler.transform(X).astype(np.float32)
    keep = (sub.prn != SUSPECT_PRN).values

    rows = []
    for disp, stem in SELECTED_CLASSICAL.items():
        p = MODEL_ROOT / "classical" / f"{stem}.joblib"
        if not p.exists():
            continue
        m = joblib.load(p)
        tau = threshold_for_recall(yv, proba(m, Xv))
        rows.append((disp, 'classical', (proba(m, X) >= tau).astype(int)))
        print(f"  scored {disp}")

    for disp, (Cls, cfg_name) in DL_REGISTRY.items():
        pt = MODEL_ROOT / "deep_learning" / f"{cfg_name}.pt"
        if not pt.exists():
            continue
        cfg = get_config(cfg_name); cfg['input_dim'] = Xv_s.shape[1]
        m = Cls(input_dim=Xv_s.shape[1], config=cfg); m.build_model()
        m.model.load_state_dict(torch.load(str(pt), map_location='cpu'))
        m.model.eval(); m.is_trained = True
        tau = threshold_for_recall(yv, proba(m, Xv_s))
        rows.append((disp, 'deep', (proba(m, Xs) >= tau).astype(int)))
        print(f"  scored {disp}")

    out = []
    for disp, fam, flag in rows:
        d_all = pd.DataFrame({'p': sub.point, 'f': flag}).groupby('p').f.mean()
        d_ex = (pd.DataFrame({'p': sub.point[keep], 'f': flag[keep]})
                  .groupby('p').f.mean())
        out.append(dict(detector=disp, family=fam,
                        evaded_all=int((d_all < EVADE_BELOW).sum()),
                        evaded_no_prn13=int((d_ex < EVADE_BELOW).sum()),
                        worst_all=round(float(d_all.min()), 3),
                        worst_no_prn13=round(float(d_ex.min()), 3)))
    df = pd.DataFrame(out).sort_values('evaded_all', ascending=False)
    df['delta'] = df.evaded_no_prn13 - df.evaded_all
    df.to_csv(OUT / 's3_prn13_check.csv', index=False)
    print("\n" + df.to_string(index=False))
    print("\nIf 'delta' is small, the result does not rest on PRN 13.")


if __name__ == "__main__":
    main()
