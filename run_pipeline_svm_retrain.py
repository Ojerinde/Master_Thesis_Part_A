"""
Targeted rerun: retrain ONLY the SVM detector (now with proper GridSearchCV
tuning, which the original run's SVM never got -- see
experiments/01b_retrain_svm_only.py's docstring), then rerun every downstream
stage so tables/figures reflect it. Reuses the other 13 detectors -- 7 other
classical models + 6 deep learning models -- as-is from a prior full run
instead of paying for a full 01+02 retrain (~6-7h of already-correct
training).

PREREQUISITE: results/models/classical/*.joblib and
results/models/deep_learning/*.pt from a prior full run (e.g. downloaded from
the bn_tok Kaggle run and unzipped into place) must already be present for
the OTHER 13 detectors before running this. Checked explicitly below and
FAILS LOUDLY if anything is missing, rather than letting the downstream
stages silently skip a missing model (their loops print "MISSING <name>" and
continue -- see e.g. experiments/12_operating_point.py -- which would produce
a quietly-incomplete run, not an error).

Usage (from the repo root):
    python run_pipeline_svm_retrain.py
"""
import os
import sys
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ['PYTHONPATH'] = str(ROOT)
sys.path.insert(0, str(ROOT))

from config.paths import CLASSICAL_MODELS, DL_MODELS  # noqa: E402

# The 7 OTHER classical stems every downstream stage selects (SVM itself is
# the 8th -- produced fresh by 01b, not checked here) and the 6 DL stems.
# Kept in sync with SELECTED_CLASSICAL / CLASSICAL / DL_REGISTRY / DL across
# 12/13/14/17/18/19/20/22 -- all eight dicts list the same 8 classical + 6 DL
# names (verified by grep across experiments/ when this script was written).
REQUIRED_CLASSICAL = [
    'RandomForest_default', 'XGBoost_default', 'LightGBM_default',
    'GradientBoosting', 'KNN', 'MLP', 'DecisionTree',
]
REQUIRED_DL = ['cnn_1d', 'lstm', 'bilstm', 'cnn_lstm', 'transformer', 'tcn']

STEPS = [
    ('01b_retrain_svm_only.py',       'retrain + GridSearchCV-tune ONLY the SVM detector'),
    ('12_operating_point.py',         'common recall-0.95 operating point (tau*)'),
    ('13_blackbox_attacks.py',        'decision-based boundary attack (gradient-masking-proof)'),
    ('25_fragility_ci.py',            'bootstrap CI on the fragility ranking (Fig 6/7 error bars)'),
    ('14_multisurrogate_transfer.py', 'multi-surrogate transfer attack'),
    ('18_full_eval.py',               'FGSM/PGD/DLSA/SNA/TPA master table at tau*'),
    ('17_latency.py',                 'inference + attack-generation latency'),
    ('20_mono.py',                    'PGD monotonicity diagnostic'),
    ('19_final_analysis.py',          'Table-1 CIs, worst-case, Mann-Whitney'),
    ('22_mcnemar.py',                 'McNemar comparability'),
]


def preflight():
    missing = []
    for stem in REQUIRED_CLASSICAL:
        p = CLASSICAL_MODELS / f"{stem}.joblib"
        if not p.exists():
            missing.append(str(p))
    for stem in REQUIRED_DL:
        p = DL_MODELS / f"{stem}.pt"
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("PREFLIGHT FAILED -- the other 13 detectors must already be "
              "in place before a targeted SVM-only rerun (this script does "
              "not train them). Missing:")
        for m in missing:
            print(f"    {m}")
        print("\nUnzip a prior full run's results/ (e.g. "
              "results_minmax_bn_tok.zip from the Kaggle core-pipeline run) "
              "into place first -- see KAGGLE_RUNBOOK.md, 'Retrain SVM only.'")
        sys.exit(2)
    print(f"Preflight OK -- all {len(REQUIRED_CLASSICAL)} other classical + "
          f"{len(REQUIRED_DL)} deep learning model files are present.")


def main():
    preflight()
    py = sys.executable
    t0 = time.time()
    for i, (script, desc) in enumerate(STEPS, 1):
        path = ROOT / 'experiments' / script
        print(f"\n{'='*74}\n[{i:2d}/{len(STEPS)}] {script}  --  {desc}\n{'='*74}",
              flush=True)
        if not path.exists():
            print(f"  MISSING {path} -- stopping."); sys.exit(2)
        s = time.time()
        r = subprocess.run([py, str(path)], cwd=str(ROOT),
                           env=dict(os.environ, PYTHONUTF8='1'))
        if r.returncode != 0:
            print(f"\nFAILED at {script} (exit {r.returncode}). Stopping so "
                  f"the error is not masked by later stages.")
            sys.exit(r.returncode)
        print(f"  {script} done in {time.time()-s:.1f}s")
    print(f"\n{'='*74}\nSVM RETRAIN + DOWNSTREAM RERUN COMPLETE in "
          f"{time.time()-t0:.1f}s\nTables -> results/tables/   "
          f"Figures -> results/figures/\n{'='*74}")


if __name__ == '__main__':
    main()
