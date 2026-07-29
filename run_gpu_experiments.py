"""
GPU experiment pipeline for Paper 1 - the long unattended runs.
================================================================
Runs the two remaining heavy experiments end to end so they can go overnight on a
Kaggle GPU (or locally):

  23_generalization.py  full cross-scenario + leave-PRN, all 14 models  -> generalization.csv
  24_defense.py         diagnostic adversarial-training baseline (6 DL)  -> defense_baseline.csv

It SMOKE-TESTS each stage first (a 1-fold / few-epoch / subsampled pass) so an error
surfaces in minutes instead of after hours. Only if every smoke passes does it start
the full runs. Smoke output files are REMOVED once the smoke passes, so no smoke
artifact is left behind. Fail-fast; every stage streams to the console and to a log
under results/tables/_gpu_logs/.

Cleanup scope: this removes ONLY the two Paper-1 smoke files it creates
(generalization_smoke.csv, defense_baseline_smoke.csv). It deliberately does NOT touch
Paper-2 smoke files (gnss_shield_*_smoke, wp2_*_smoke, certification_*_smoke, ...),
which live in the same gitignored folder but belong to the other branch.

Run:
    python run_gpu_experiments.py                # smoke, then full (overnight)
    python run_gpu_experiments.py --smoke-only   # just the fast checks (no long run)
    python run_gpu_experiments.py --skip-smoke   # full only (smoke already verified)
"""
import os
import sys
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ['PYTHONPATH'] = str(ROOT)
sys.path.insert(0, str(ROOT))
from config.paths import TABLES_DIR                             # noqa: E402

LOG_DIR = TABLES_DIR / '_gpu_logs'
# ONLY the smoke files these stages create. Not Paper-2 smoke files.
SMOKE_ARTIFACTS = ['generalization_smoke.csv', 'defense_baseline_smoke.csv']


def run(script, args, desc, i, n, tag):
    path = ROOT / 'experiments' / script
    print(f"\n{'='*74}\n[{tag} {i}/{n}] {script} {' '.join(args)}  --  {desc}\n{'='*74}",
          flush=True)
    if not path.exists():
        print(f"  MISSING {path} -- stopping."); sys.exit(2)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"{tag.lower()}_{i:02d}_{script.replace('.py', '')}.log"
    s = time.time()
    with open(log, 'w', encoding='utf-8') as lf:
        lf.write(f"# {script} {args} -- {datetime.now().isoformat()}\n\n"); lf.flush()
        p = subprocess.Popen(
            [sys.executable, '-u', str(path), *args], cwd=str(ROOT),
            env=dict(os.environ, PYTHONWARNINGS='ignore', PYTHONUTF8='1'),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            encoding='utf-8')
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush(); lf.write(line)
        p.wait()
    dt = time.time() - s
    if p.returncode != 0:
        print(f"\nFAILED at {script} {args} (exit {p.returncode}) after {dt:.1f}s. "
              f"See {log}. Stopping so the error is not masked.")
        sys.exit(p.returncode)
    print(f"  {script} done in {dt:.1f}s  (log: {log.name})")


def cleanup_smoke():
    print(f"\n{'='*74}\nRemoving Paper-1 smoke artifacts\n{'='*74}")
    for f in SMOKE_ARTIFACTS:
        p = TABLES_DIR / f
        if p.exists():
            p.unlink(); print(f"  removed {p.name}")
        else:
            print(f"  (absent) {f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-smoke', action='store_true',
                    help='skip the smoke checks and run the full stages directly')
    ap.add_argument('--smoke-only', action='store_true',
                    help='run only the smoke checks (fast error check), then stop')
    ap.add_argument('--with-defense', action='store_true',
                    help='also run the diagnostic defense baseline (24). OFF by '
                         'default: generalization (23) is the solid, low-risk run; '
                         'the defense outcome needs review before inclusion.')
    ap.add_argument('--protocol', choices=['all', 'cross_scenario', 'leave_prn'],
                    default='all',
                    help='forwarded to 23_generalization.py -- run one protocol '
                         'per Kaggle session so each commit completes inside the '
                         '12h cap (a timeout kill does NOT preserve any output; '
                         'a completed run does). generalization.csv accumulates '
                         'across sessions via its own resume/skip logic.')
    ap.add_argument('--batch-size', type=int, default=None,
                    help='forwarded to 23_generalization.py and 24_defense.py -- '
                         'DL batch size (config default 32 under-uses a GPU)')
    ap.add_argument('--models', default=None,
                    help='forwarded to 24_defense.py -- comma-separated subset of '
                         'the 6 DL models, to chunk the defense run across sessions')
    args = ap.parse_args()
    t0 = time.time()

    gen_full_args = [] if args.protocol == 'all' else ['--protocol', args.protocol]
    def_full_args = []
    if args.batch_size:
        gen_full_args += ['--batch-size', str(args.batch_size)]
        def_full_args += ['--batch-size', str(args.batch_size)]
    if args.models:
        def_full_args += ['--models', args.models]

    smoke_core = [('23_generalization.py', ['--smoke'], 'generalization smoke (1+1 folds)')]
    smoke_def = [('24_defense.py', ['--smoke'], 'defense smoke (2 DL models)')]
    full_core = [('23_generalization.py', gen_full_args,
                  f'generalization (protocol={args.protocol}) -> generalization.csv')]
    full_def = [('24_defense.py', def_full_args,
                 'diagnostic defense baseline -> defense_baseline.csv')]

    smoke_steps = smoke_core + (smoke_def if args.with_defense else [])
    full_steps = full_core + (full_def if args.with_defense else [])

    if not args.skip_smoke:
        for i, (s, a, d) in enumerate(smoke_steps, 1):
            run(s, a, d, i, len(smoke_steps), 'SMOKE')
        cleanup_smoke()
        if args.smoke_only:
            print(f"\nSmoke-only complete in {time.time()-t0:.1f}s. No errors.")
            return
        print("\nSmoke passed. Starting the full runs.")

    for i, (s, a, d) in enumerate(full_steps, 1):
        run(s, a, d, i, len(full_steps), 'FULL')

    print(f"\n{'='*74}\nGPU PIPELINE COMPLETE in {time.time()-t0:.1f}s\n"
          f"Tables -> {TABLES_DIR}\nLogs   -> {LOG_DIR}\n{'='*74}")


if __name__ == '__main__':
    main()
