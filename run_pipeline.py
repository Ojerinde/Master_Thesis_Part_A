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
# 18, since 18 merges 13's boundary CSV) -> stats (19,22).
#
# 21_figures.py and 16_revision_figures.py are NOT run here. They are legacy,
# from the earlier GPS Solutions submission (figure names like
# fig01_clean_performance.png, see manuscript/GPS_Solutions_sn/figures/), write
# to a hardcoded Windows path (papers/_archive/GPS_Solutions_sn/figures) that
# has nothing to do with the current paper and would likely fail on a Linux
# host (Kaggle), aborting the run at the last stage after 01/02 already spent
# the GPU time. The current, live figure generator is
# papers/paper1-satnav/make_figures.py, which reads directly from this
# pipeline's results/tables/ output; run it separately after this completes.
STEPS = [
    ('01_classical_baseline.py',      'train 7 classical detectors (+variants)'),
    ('02_deep_learning_baseline.py',  'train 6 deep detectors'),
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
        # PYTHONUTF8=1: several stages print non-ASCII (checkmarks etc.); when
        # stdout is redirected (a log file, a pipe, Kaggle's captured output)
        # Windows falls back to the cp1252 console codepage and crashes with
        # UnicodeEncodeError on those characters. Force UTF-8 mode (PEP 540)
        # regardless of platform or whether stdout is a real console.
        r = subprocess.run([py, str(path)], cwd=str(ROOT),
                           env=dict(os.environ, PYTHONUTF8='1'))
        if r.returncode != 0:
            print(f"\nFAILED at {script} (exit {r.returncode}). Stopping so the "
                  f"error is not masked by later stages.")
            sys.exit(r.returncode)
        print(f"  {script} done in {time.time()-s:.1f}s")
    print(f"\n{'='*74}\nPIPELINE COMPLETE in {time.time()-t0:.1f}s\n"
          f"Tables -> results/tables/   Figures -> results/figures/\n{'='*74}")


if __name__ == '__main__':
    main()
