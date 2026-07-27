"""
GNSS Adversarial ML Research — LEGACY pipeline entry point
====================================================
Superseded by run_pipeline.py, which runs the current scripts (01, 02, then
12/13/14/18/17/20/19/22/21/16) in the min-max [0,1] feature space that the
Paper-1 manuscript numbers come from. This file still works, but Stages 3 and
4 below (03_adversarial_evaluation, 04_statistical_analysis) are the earlier,
StandardScaler-era analysis and are NOT what produced the reported results.
Kept for anyone who wants that earlier flow specifically.

Runs the complete experiment pipeline:
  Stage 1: Classical baseline training    (01_classical_baseline)
  Stage 2: Deep learning baseline training (02_deep_learning_baseline)
  Stage 3: Adversarial robustness evaluation (03_adversarial_evaluation) [legacy]
  Stage 4: Statistical analysis            (04_statistical_analysis) [legacy]
  Stage 5: Manuscript figure generation    (05_manuscript_figures) [legacy]

Usage:
    python main.py                # full pipeline (all 5 stages)
    python main.py --stage 1      # classical baselines only
    python main.py --stage 1 2    # classical + DL baselines
    python main.py --stage 3 --quick  # adversarial eval, fast mode
    python main.py --stage 5      # regenerate all manuscript figures
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

STAGES = {
    1: ("Classical Baseline Training", "experiments.01_classical_baseline"),
    2: ("Deep Learning Baseline Training", "experiments.02_deep_learning_baseline"),
    3: ("Adversarial Robustness Evaluation", "experiments.03_adversarial_evaluation"),
    4: ("Statistical Analysis", "experiments.04_statistical_analysis"),
    5: ("Manuscript Figure Generation", "experiments.05_manuscript_figures"),
}


def run_stage(stage_num, module_name, label, extra_args=None):
    """Run one experiment stage as a subprocess."""
    print(f"\n{'=' * 70}")
    print(f"  STAGE {stage_num}: {label}")
    print(f"{'=' * 70}")

    cmd = [sys.executable, "-m", module_name]
    if extra_args:
        cmd.extend(extra_args)

    print(f"  Command: {' '.join(cmd)}")
    start = time.time()

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    elapsed = time.time() - start
    mins, secs = divmod(int(elapsed), 60)
    hrs, mins = divmod(mins, 60)

    if result.returncode != 0:
        print(f"\n  STAGE {stage_num} FAILED (exit code {result.returncode}) "
              f"after {hrs:02d}:{mins:02d}:{secs:02d}")
        sys.exit(result.returncode)

    print(
        f"\n  Stage {stage_num} completed in {hrs:02d}:{mins:02d}:{secs:02d}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="GNSS Adversarial ML Research Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage", type=int, nargs="+", choices=[1, 2, 3, 4, 5],
        help="Run specific stages (default: all 1-5 in order)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Pass --quick to Stage 3 (use 2000 test samples for fast run)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stages = args.stage or [1, 2, 3, 4, 5]

    print("\n" + "=" * 70)
    print("  GNSS ADVERSARIAL ML RESEARCH PIPELINE")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Stages  : {stages}")
    print("=" * 70)

    pipeline_start = time.time()

    for s in sorted(stages):
        label, module = STAGES[s]
        extra = []
        if s == 3 and args.quick:
            extra.append("--quick")
        run_stage(s, module, label, extra_args=extra or None)

    elapsed = time.time() - pipeline_start
    mins, secs = divmod(int(elapsed), 60)
    hrs, mins = divmod(mins, 60)

    print("\n" + "=" * 70)
    print(
        f"  ALL STAGES COMPLETE — Total time: {hrs:02d}:{mins:02d}:{secs:02d}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
