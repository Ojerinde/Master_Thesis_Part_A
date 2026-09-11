"""
S3: attack the 14 detectors over the waveform-level realizable library.

Each library point is a spoofer transmitter configuration whose observables were
produced by FGI-GSRx tracking a synthesized, emittable waveform. Every epoch in a
point is spoofed, so a detector's detection rate on a point is its spoof recall
against that attack. A point evades a detector when that recall falls below the
detector's clean operating recall.

Success is evasion AND effect. A weak, code-aligned spoofer that pulls nothing
evades every detector and achieves nothing; scoring it as a win would repeat, in
signal space, the mistake the feature-space evaluation made. So a point counts as
a successful attack on a detector only when the detector misses it AND it induces
a non-trivial range error, which the pull-off rate fixes analytically at 293.05 m
per chip per second.

Operating point, split and preprocessing are the shared Paper-1 ones
(load_track_splits, block-temporal, MinMaxScaler; threshold at clean-validation
recall 0.95), identical to 12_operating_point.

Outputs (results/tables/):
  s3_detection_per_point.csv   detector x point detection rate + transmit cost
  s3_summary.csv / .md         per-detector worst-case and best undetected attack
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

# The 14-detector roster lives in the gnss_adversarial_research tree.
MODEL_ROOT = Path("D:/BEIHANG UNIVERSITY/Research/code/gnss_adversarial_research/"
                  "results/models")
CLASSICAL = MODEL_ROOT / "classical"
DL = MODEL_ROOT / "deep_learning"
LIB = Path("D:/BEIHANG UNIVERSITY/Research/data/FGI_Data/composite/"
           "realizable_library.csv")
OUT = Path(__file__).resolve().parents[1] / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

FEATS = ['cn0_dbhz', 'mean_cn0_dbhz', 'noise_cn0', 'doppler_hz',
         'i_prompt', 'q_prompt', 'dll_discr', 'pll_lock', 'fll_lock']
TARGET_RECALL = 0.95
# A point "evades" a detector when the detector flags fewer than this fraction of
# its (all-spoofed) epochs. Half is the point where the detector misses the
# majority of the attack.
EVADE_BELOW = 0.50

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


def threshold_for_recall(y_val, p_val, target=TARGET_RECALL):
    """Largest threshold whose clean-validation recall meets the floor."""
    best_tau, best_r = None, -1.0
    for tau in np.unique(p_val)[::-1]:
        r = recall_score(y_val, (p_val >= tau).astype(int), zero_division=0)
        if r >= target and (best_tau is None or tau > best_tau):
            best_tau = tau
        if r > best_r:
            best_r = r
    return (float(best_tau), True) if best_tau is not None else (0.0, False)


def main():
    print("Loading block-temporal split (MinMaxScaler)...")
    (_Xtr, Xv_eng, _Xte, _ytr, yv, _yte, feat_names, scaler) = \
        load_track_splits(verbose=False)
    Xv_eng = Xv_eng.astype(np.float64)
    Xv_scaled = scaler.transform(Xv_eng).astype(np.float32)
    assert list(feat_names) == FEATS, f"feature order mismatch: {feat_names}"

    print(f"Loading library {LIB.name}...")
    lib = pd.read_csv(LIB)
    Xlib_eng = lib[FEATS].values.astype(np.float64)
    Xlib_scaled = scaler.transform(Xlib_eng).astype(np.float32)
    # Per-point transmit cost (one row per point).
    cost = (lib.groupby('point')
               .agg(family=('family', 'first'), alpha_db=('alpha_db', 'first'),
                    phi_deg=('phi_deg', 'first'), pulloff_mps=('pulloff_mps', 'first'),
                    align_frac=('align_frac', 'first'), n=('point', 'size'))
               .reset_index())
    print(f"  {len(lib):,} rows, {cost.shape[0]} points "
          f"({(cost.family=='inphase').sum()} inphase, "
          f"{(cost.family=='neutral').sum()} neutral)")

    detectors = []  # (name, family, scores_over_library, tau)
    print("\nFixing operating points and scoring the library:")
    for disp, stem in SELECTED_CLASSICAL.items():
        p = CLASSICAL / f"{stem}.joblib"
        if not p.exists():
            print(f"  MISSING {disp}"); continue
        mdl = joblib.load(p)
        tau, feas = threshold_for_recall(yv, proba(mdl, Xv_eng))
        s = proba(mdl, Xlib_eng)
        detectors.append((disp, 'classical', s, tau))
        print(f"  {disp:16s} tau={tau:.3f} feasible={feas}")

    for disp, (Cls, cfg_name) in DL_REGISTRY.items():
        pt = DL / f"{cfg_name}.pt"
        if not pt.exists():
            print(f"  MISSING {disp}"); continue
        cfg = get_config(cfg_name); cfg['input_dim'] = Xv_scaled.shape[1]
        m = Cls(input_dim=Xv_scaled.shape[1], config=cfg); m.build_model()
        m.model.load_state_dict(torch.load(str(pt), map_location='cpu'))
        m.model.eval(); m.is_trained = True
        tau, feas = threshold_for_recall(yv, proba(m, Xv_scaled))
        s = proba(m, Xlib_scaled)
        detectors.append((disp, 'deep', s, tau))
        print(f"  {disp:16s} tau={tau:.3f} feasible={feas}")

    # Detection rate per (detector, point): fraction of the point's spoofed epochs
    # the detector flags at its operating threshold.
    pt_index = lib['point'].values
    per_rows, summ_rows = [], []
    for disp, fam, s, tau in detectors:
        flagged = (s >= tau).astype(int)
        dfp = pd.DataFrame({'point': pt_index, 'flag': flagged})
        det = dfp.groupby('point').flag.mean().reset_index(name='detection_rate')
        det = det.merge(cost, on='point')
        det['detector'] = disp; det['detector_family'] = fam
        per_rows.append(det)

        evaded = det[det.detection_rate < EVADE_BELOW]
        # Success = evaded AND non-trivial effect. Rank undetected points by the
        # range error they induce (pull-off), report the strongest and its cost.
        eff = evaded[evaded.pulloff_mps > 0]
        best = eff.sort_values('pulloff_mps', ascending=False).head(1)
        summ_rows.append(dict(
            detector=disp, family=fam, tau=round(float(tau), 4),
            min_detection=round(float(det.detection_rate.min()), 3),
            median_detection=round(float(det.detection_rate.median()), 3),
            n_points_evaded=int(len(evaded)),
            n_evaded_with_effect=int(len(eff)),
            max_undetected_pulloff_mps=round(float(best.pulloff_mps.iloc[0]), 1)
                if len(best) else 0.0,
            best_attack_family=best.family.iloc[0] if len(best) else '',
            best_attack_alpha_db=round(float(best.alpha_db.iloc[0]), 2)
                if len(best) else np.nan,
            best_attack_phi_deg=int(best.phi_deg.iloc[0]) if len(best) else 0,
            best_attack_point=best.point.iloc[0] if len(best) else '',
        ))

    per = pd.concat(per_rows, ignore_index=True)
    per.to_csv(OUT / 's3_detection_per_point.csv', index=False)
    summ = pd.DataFrame(summ_rows).sort_values('max_undetected_pulloff_mps',
                                               ascending=False)
    summ.to_csv(OUT / 's3_summary.csv', index=False)
    with open(OUT / 's3_summary.md', 'w') as f:
        f.write("# S3: realizable-library attack on 14 detectors\n\n")
        f.write("Detection rate = spoof recall on the attack; a point evades when "
                f"below {EVADE_BELOW}. Success = evaded AND non-trivial pull-off.\n\n")
        f.write(summ.to_markdown(index=False))
        f.write("\n")
    print("\n" + summ.to_string(index=False))
    print(f"\nWrote {OUT / 's3_summary.csv'}")


if __name__ == "__main__":
    main()
