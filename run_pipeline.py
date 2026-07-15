"""
One-command runner for the Paper-1 pipeline (FGI-GSRx corpus, Satellite Navigation).

Runs every stage in dependency order and STOPS on the first failure, so a smoke
run on the current corpus (or the full run on the complete corpus) is a single
command. The scripts are run as files (their names start with digits, so
`python -m experiments.NN_...` is not importable); each sets PYTHONPATH itself,
but we also export it for safety.

Usage (from the repo root):
    python run_pipeline.py

Prereq: the corpus data/processed/texbat_track_combined.csv exists (produced by
fgi/export_texbat_track.m). Models are trained by steps 01-02; everything after
consumes them. Re-run after ds2/ds3 are added to the corpus for the final numbers.
"""
import os
import sys
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ['PYTHONPATH'] = str(ROOT)

# Dependency order: train (01,02) -> operating point (12) -> attacks (13 before
# 18, since 18 merges 13's boundary CSV) -> stats (19,22) -> figures (21,16).
STEPS = [
    ('01_classical_baseline.py',      'train 7 classical detectors (+variants)'),
    ('02_deep_learning_baseline.py',  'train 6 deep detectors'),
    ('12_operating_point.py',         'common recall-0.95 operating point (tau*)'),
    ('13_blackbox_attacks.py',        'decision-based boundary attack (gradient-masking-proof)'),
    ('14_multisurrogate_transfer.py', 'multi-surrogate transfer attack'),
    ('18_full_eval.py',               'FGSM/PGD/DLSA/SNA/TPA master table at tau*'),
    ('17_latency.py',                 'inference + attack-generation latency'),
    ('20_mono.py',                    'PGD monotonicity diagnostic'),
    ('19_final_analysis.py',          'Table-1 CIs, worst-case, Mann-Whitney'),
    ('22_mcnemar.py',                 'McNemar comparability'),
    ('21_figures.py',                 'data-driven manuscript figures'),
    ('16_revision_figures.py',        'revision figures'),
]


def main():
    py = sys.executable
    t0 = time.time()
    for i, (script, desc) in enumerate(STEPS, 1):
        path = ROOT / 'experiments' / script
        print(f"\n{'='*74}\n[{i:2d}/{len(STEPS)}] {script}  --  {desc}\n{'='*74}",
              flush=True)
        if not path.exists():
            print(f"  MISSING {path} -- stopping."); sys.exit(2)
        s = time.time()
        r = subprocess.run([py, str(path)], cwd=str(ROOT))
        if r.returncode != 0:
            print(f"\nFAILED at {script} (exit {r.returncode}). Stopping so the "
                  f"error is not masked by later stages.")
            sys.exit(r.returncode)
        print(f"  {script} done in {time.time()-s:.1f}s")
    print(f"\n{'='*74}\nPIPELINE COMPLETE in {time.time()-t0:.1f}s\n"
          f"Tables -> results/tables/   Figures -> results/figures/\n{'='*74}")


if __name__ == '__main__':
    main()
