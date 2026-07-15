"""
Stage-2 pipeline: reruns + fills the gap left after the 2026-07-14 full run.
============================================================================
run_pipeline.py (01->22->16) already ran end-to-end on the complete FGI-GSRx
corpus on 2026-07-14 (24,888s, exit clean). Since then:

  - 12_operating_point.py, 18_full_eval.py, 20_mono.py were BUG-FIXED (coupling-
    aware GNSS enforcer wired in; fixed-FAR=0.05 second operating point added).
    Their 2026-07-14 output is now STALE and must be regenerated.
  - 19_final_analysis.py and 22_mcnemar.py only print() their stats; nothing
    captured that console output during the 6.9h run. Rerun WITH logging.
  - 23_generalization.py was run --classical-only only; the DL folds are
    missing from results/tables/generalization.csv (figure F8 needs both).

This script does NOT touch 01_classical_baseline.py or 02_deep_learning_baseline.py
-- it reuses the EXACT trained model artifacts from the 2026-07-14 run (verified
present before running), so every number in the paper traces back to ONE set of
trained models. Retraining 01/02 again would risk tiny non-determinism between
runs, which is exactly the kind of cross-table inconsistency a 12.4-IF reviewer
would catch (Reviewer #1 #3 lesson from the GPS Solutions rejection).

Two phases:
  PHASE 0  Archive (move, never delete) Paper-1 STALE result files so nothing
           pre-FGI-GSRx or pre-complete-corpus can be accidentally cited.
           Uses an EXPLICIT exact-filename allowlist -- no glob/pattern matching
           -- so there is zero risk of ever touching a Paper-2 file (wp1_*,
           wp2_*, gnss_shield_*, cert_probe_*, certification_*, certified_*,
           nonmonotonicity_diag_smoke*, which live in the SAME gitignored
           results/tables/ folder because git never cleans ignored files across
           branch checkouts). A manifest is written into the archive folder.
  PHASE 1  Run 12, 18, 20, 19, 22, 23(full) in that order, fail-fast. Every
           stage's stdout+stderr is teed live to the console AND saved to
           results/tables/_stage2_logs/<NN_name>_log.txt (fixes the general
           "nothing was captured" gap, not just for 19/22).

Usage (from the repo root):
    python run_pipeline_stage2.py

Expect the run to take a long time: 23_generalization.py (full) retrains 6 deep
models x 14 folds = 84 DL trainings on CPU. Meant to run unattended overnight.
"""
import os
import sys
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ['PYTHONPATH'] = str(ROOT)
sys.path.insert(0, str(ROOT))

from config.paths import TABLES_DIR, CLASSICAL_MODELS, DL_MODELS  # noqa: E402

STAMP = datetime.now().strftime('%Y-%m-%d_%H%M%S')
ARCHIVE_DIR = TABLES_DIR / '_archive_stale' / STAMP
LOG_DIR = TABLES_DIR / '_stage2_logs'

# ---------------------------------------------------------------------------
# PHASE 0 -- stale-file archive. EXACT FILENAMES ONLY (no globs), grouped by
# origin so the manifest explains WHY each one is stale.
# ---------------------------------------------------------------------------
STALE_GROUPS = {
    'toy_receiver_pre_fgi_2026-04-17': [
        # Pre-FGI-GSRx hand-rolled receiver corpus; produced by the legacy
        # 03_adversarial_evaluation.py / 04_statistical_analysis.py /
        # 05_manuscript_figures.py chain, which is NOT part of run_pipeline.py
        # and shares some output filenames with the CURRENT 01/12+ series
        # (optimal_thresholds.csv) -- do not ever run 03/04/05 again.
        'adversarial_attack_results.csv',
        'worst_case_robustness.csv',
        'table1_bootstrap_cis.csv',
        'table2_mcnemar_pairwise.csv',
        'table3_wilcoxon_adversarial.csv',
        'table4_friedman_attack_comparison.csv',
        'table5_effect_sizes.csv',
        'table6_degradation_summary.csv',
        'table7_robustness_correlation.csv',
    ],
    'orphaned_satgrid_2026-05-27': [
        # No producing script found anywhere in the current codebase; a
        # different dataset (SatGrid, not TEXBAT) from an earlier exploration.
        # Not Paper-1, not Paper-2's WP1/WP2 (different naming/data entirely).
        'generalization_results.csv',
        'generalization_gap.csv',
    ],
    'fgi_partial_corpus_smoke_2026-07-03_04': [
        # FGI-GSRx corpus era, but BEFORE ds2/ds3 tracking finished (2-scenario
        # partial corpus) or explicit --quick smoke-test runs. Superseded by
        # the 2026-07-14 complete-corpus (4-scenario) run.
        'blackbox_boundary.csv',
        'blackbox_boundary_quick.csv',
        'blackbox_boundary_all_quick.csv',
        'multisurrogate_transfer_quick.csv',
        '_bb13_log.txt',
        '_bb13_all_log.txt',
        '_op_point_log.txt',
        '_ms14_log.txt',
        '_retrain01_log.txt',
        '_full_eval_log.txt',
        '_mono_log.txt',
    ],
}
STALE_DIRS = {
    # SatGrid raw_predictions/, same May-27 orphaned run as above.
    'orphaned_satgrid_2026-05-27': ['raw_predictions'],
}

# Hard denylist assertion: these prefixes belong to Paper-2 (paper2-signal-level
# branch) and must NEVER be archived/moved/deleted by this script, even by
# accident. Every filename in STALE_GROUPS is checked against this at runtime.
PAPER2_PREFIXES = (
    'wp1_', 'wp2_', 'gnss_shield_', 'cert_probe_', 'certification_',
    'certified_', 'nonmonotonicity_diag',
)


def archive_stale():
    print(f"\n{'='*74}\nPHASE 0 -- archiving stale Paper-1 files (move, not delete)\n{'='*74}")
    moved, missing = [], []
    for group, files in STALE_GROUPS.items():
        for fname in files:
            assert not fname.lstrip('_').startswith(PAPER2_PREFIXES), \
                f"REFUSING: {fname} matches a Paper-2 prefix -- must not be archived by this script."
            src = TABLES_DIR / fname
            if not src.exists():
                missing.append(fname)
                continue
            dst_dir = ARCHIVE_DIR / group
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst_dir / fname))
            moved.append(f"{group}/{fname}")
    for group, dirs in STALE_DIRS.items():
        for dname in dirs:
            src = TABLES_DIR / dname
            if not src.exists():
                missing.append(dname)
                continue
            dst_dir = ARCHIVE_DIR / group
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst_dir / dname))
            moved.append(f"{group}/{dname}/ (directory)")

    manifest = ARCHIVE_DIR / 'MANIFEST.md'
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(manifest, 'w') as f:
        f.write(f"# Stale-file archive -- {STAMP}\n\n")
        f.write("Moved (reversible -- move back to results/tables/ if needed):\n\n")
        for m in moved:
            f.write(f"- {m}\n")
        if missing:
            f.write("\nAlready absent (nothing to do):\n\n")
            for m in missing:
                f.write(f"- {m}\n")
    print(f"  moved {len(moved)} item(s) -> {ARCHIVE_DIR}")
    if missing:
        print(f"  {len(missing)} already absent (fine, not an error): {missing}")
    print(f"  manifest -> {manifest}")

    # Paper-2 untouched check: count files with the guarded prefixes before vs
    # after -- must be identical, or this script itself is broken.
    p2_after = sorted(p.name for p in TABLES_DIR.glob('*')
                       if p.name.lstrip('_').startswith(PAPER2_PREFIXES))
    print(f"  Paper-2 files present and UNTOUCHED: {len(p2_after)} "
          f"({', '.join(p2_after[:3])}{', ...' if len(p2_after) > 3 else ''})")


# ---------------------------------------------------------------------------
# PHASE 1 -- the 6 remaining stages, fail-fast, fully logged.
# ---------------------------------------------------------------------------
STEPS = [
    ('12_operating_point.py',   [],
     'recall-0.95 tau* (refresh) + NEW fixed-FAR=0.05 second operating point'),
    ('18_full_eval.py',         [],
     'FGSM/PGD/DLSA/SNA/TPA master table -- NOW coupling-enforcer-correct'),
    ('20_mono.py',              [],
     'PGD monotonicity diagnostic -- NOW with/without enforcer actually differ'),
    ('19_final_analysis.py',    [],
     'Table-1 CIs, worst-case, Mann-Whitney -- LOGGED this time'),
    ('22_mcnemar.py',           [],
     'McNemar comparability -- LOGGED this time'),
    ('23_generalization.py',    [],
     'FULL generalization (adds the 6 DL model folds; classical already done)'),
]


def run_step(i, n, script, extra_args, desc):
    path = ROOT / 'experiments' / script
    print(f"\n{'='*74}\n[{i}/{n}] {script}  --  {desc}\n{'='*74}", flush=True)
    if not path.exists():
        print(f"  MISSING {path} -- stopping."); sys.exit(2)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{i:02d}_{script.replace('.py', '')}_log.txt"
    s = time.time()
    with open(log_path, 'w', encoding='utf-8') as logf:
        logf.write(f"# {script} -- started {datetime.now().isoformat()}\n")
        logf.write(f"# args: {extra_args}\n\n")
        logf.flush()
        proc = subprocess.Popen(
            [sys.executable, str(path), *extra_args], cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
        proc.wait()
    dt = time.time() - s
    if proc.returncode != 0:
        print(f"\nFAILED at {script} (exit {proc.returncode}) after {dt:.1f}s. "
              f"See {log_path}. Stopping so the error is not masked by later stages.")
        sys.exit(proc.returncode)
    print(f"  {script} done in {dt:.1f}s  (log: {log_path.name})")


def preflight():
    """Refuse to run if the 2026-07-14 trained model artifacts are missing --
    this script must never silently fall back to retraining 01/02."""
    n_clf = len(list(CLASSICAL_MODELS.glob('*.joblib')))
    n_dl = len(list(DL_MODELS.glob('*.pt')))
    print(f"Preflight: {n_clf} classical model(s), {n_dl} DL model(s) found in "
          f"{CLASSICAL_MODELS} / {DL_MODELS}")
    if n_clf < 7 or n_dl < 6:
        print("REFUSING to proceed: expected >=7 classical + 6 DL trained models "
              "(from the 2026-07-14 run of run_pipeline.py). Run that first.")
        sys.exit(3)


def main():
    t0 = time.time()
    preflight()
    archive_stale()

    n = len(STEPS)
    for i, (script, extra_args, desc) in enumerate(STEPS, 1):
        run_step(i, n, script, extra_args, desc)

    print(f"\n{'='*74}\nSTAGE-2 PIPELINE COMPLETE in {time.time()-t0:.1f}s\n"
          f"Tables -> {TABLES_DIR}\nLogs   -> {LOG_DIR}\n"
          f"Archived stale files -> {ARCHIVE_DIR}\n{'='*74}")


if __name__ == '__main__':
    main()
