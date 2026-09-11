"""
Faithful CPU retrain of the RBF-SVM detector.

The shipped SVM.joblib was trained with RAPIDS cuML on a Kaggle GPU and will not
unpickle without cuml (ModuleNotFoundError: cuml). Every other one of the 14
detectors loads locally. This rebuilds the SVM as a plain sklearn RBF-SVC so the
full roster is available on this machine for the waveform-level attack.

Faithfulness is by construction, not by hand:
  - the split is load_track_splits(), the same block-temporal purged partition
    every Paper-1 experiment uses, so train/val/test are identical to the roster;
  - the pipeline is MinMaxScaler + SVC, matching the shipped classical Pipelines
    (verified: each carries an internal MinMaxScaler) and the manuscript's
    min-max observable space;
  - the hyperparameters are the tuned values recorded for the shipped model,
    C=10.0 and gamma='scale' (baseline_results.csv), with the RBF kernel,
    class_weight='balanced', probability=True and random_state=42 from
    SVM_RBF_CONFIG.

cuML's SVC and sklearn's SVC solve the same dual QP, so at matched C, gamma and
kernel the decision boundary is the same up to solver tolerance. The one honest
difference to note in the methods text: this SVC is exact RBF, whereas any CPU
RFF-approximation fallback would not be, so the exact kernel is used here.

Run:
    python -m experiments.01c_retrain_svm_cpu
"""
import sys, time, json
from pathlib import Path
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.loader import load_track_splits

# The 14-detector roster lives in the gnss_adversarial_research tree; write there
# so the eval harness finds the retrained SVM alongside the others.
OUT = Path("D:/BEIHANG UNIVERSITY/Research/code/gnss_adversarial_research/"
           "results/models/classical")
OUT.mkdir(parents=True, exist_ok=True)

# Tuned values recorded for the shipped SVM (baseline_results.csv:
# {'model__C': 10.0, 'model__gamma': 'scale'}) plus the fixed SVM_RBF_CONFIG.
SVM_PARAMS = dict(C=10.0, kernel="rbf", gamma="scale",
                  class_weight="balanced", probability=True, random_state=42)


def op_threshold(scores, y, recall_floor=0.95):
    """Highest threshold whose recall on this block still meets the floor, the
    common operating point of Equation (oppoint) in the manuscript."""
    order = np.argsort(-scores)
    thr_grid = np.unique(scores)[::-1]
    best = 0.0
    for t in thr_grid:
        r = recall_score(y, (scores >= t).astype(int), zero_division=0)
        if r >= recall_floor:
            best = t
        else:
            break
    return best


def main():
    print("Loading Paper-1 block-temporal split (MinMaxScaler, purge=20)...")
    Xtr, Xva, Xte, ytr, yva, yte, feats, _ = load_track_splits(verbose=True)
    print(f"features ({len(feats)}): {feats}")

    pipe = Pipeline([("scaler", MinMaxScaler()),
                     ("model", SVC(**SVM_PARAMS))])
    print(f"Fitting RBF-SVC (C={SVM_PARAMS['C']}, gamma={SVM_PARAMS['gamma']}) "
          f"on {len(Xtr):,} rows; probability=True runs an internal 5-fold, so "
          f"this takes a while...")
    t = time.time()
    pipe.fit(Xtr, ytr)
    print(f"fit done in {time.time()-t:.0f} s, "
          f"{pipe.named_steps['model'].n_support_.sum()} support vectors")

    # Report at the common 0.95-recall operating point, on val and test.
    out = {"params": SVM_PARAMS, "features": feats}
    for name, X, y in [("val", Xva, yva), ("test", Xte, yte)]:
        s = pipe.predict_proba(X)[:, 1]
        eta = op_threshold(pipe.predict_proba(Xva)[:, 1], yva) if name == "test" \
            else op_threshold(s, y)
        pred = (s >= eta).astype(int)
        out[name] = dict(eta=float(eta),
                         f1=float(f1_score(y, pred, zero_division=0)),
                         recall=float(recall_score(y, pred, zero_division=0)),
                         precision=float(precision_score(y, pred, zero_division=0)),
                         auc=float(roc_auc_score(y, s)))
        print(f"  {name}: eta={eta:.4f}  F1={out[name]['f1']:.4f}  "
              f"recall={out[name]['recall']:.4f}  AUC={out[name]['auc']:.4f}")

    # Save alongside the roster. Keep the cuML original rather than overwrite it,
    # so provenance is preserved; the eval harness loads SVM.joblib.
    dst = OUT / "SVM.joblib"
    if dst.exists():
        (OUT / "SVM_cuml_original.joblib.bak").write_bytes(dst.read_bytes())
    joblib.dump(pipe, dst)
    (OUT / "SVM_cpu_retrain_metrics.json").write_text(json.dumps(out, indent=2))
    print(f"saved {dst}")
    print("The paper reports clean F1 0.682 for the SVM; compare out['test']['f1'].")


if __name__ == "__main__":
    main()
