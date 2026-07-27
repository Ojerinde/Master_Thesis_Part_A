"""
95% bootstrap CI on the decision-based boundary attack's median min-L-inf
fragility, per detector. Feeds Figure 6/7's error bars (make_figures.py::fig_f6).

Input:  results/tables/blackbox_boundary_persample.csv (13_blackbox_attacks.py)
Output: results/tables/blackbox_boundary_ci.csv

Run: PYTHONPATH=. python experiments/25_fragility_ci.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from config.paths import TABLES_DIR                             # noqa: E402

N_BOOT = 10_000
SEED = 42


def main():
    src = TABLES_DIR / 'blackbox_boundary_persample.csv'
    df = pd.read_csv(src)
    rng = np.random.default_rng(SEED)

    rows = []
    for model, g in df.groupby('model', sort=False):
        v = g['min_linf'].to_numpy(dtype=float)
        n = len(v)
        med = float(np.median(v))
        boot_meds = np.empty(N_BOOT)
        for i in range(N_BOOT):
            boot_meds[i] = np.median(rng.choice(v, size=n, replace=True))
        ci_lo, ci_hi = np.percentile(boot_meds, [2.5, 97.5])
        rows.append({'model': model, 'median_min_linf': round(med, 4),
                     'ci_lo': round(float(ci_lo), 4), 'ci_hi': round(float(ci_hi), 4),
                     'n': n})

    out = pd.DataFrame(rows).sort_values('median_min_linf')
    dst = TABLES_DIR / 'blackbox_boundary_ci.csv'
    out.to_csv(dst, index=False)
    print(f"wrote {dst}  ({len(out)} models, {N_BOOT} resamples each)")
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
