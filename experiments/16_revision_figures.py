"""
Regenerate the manuscript figures whose data changed in the no-advantage
revision, writing PNGs into the manuscript figures directory (filenames kept
stable so the .tex is unchanged).

- fig09_spearman.png  -> decision-based ASR at eps=0.10 per model, by family
                         (replaces the old Spearman scatter).
- fig01_clean_performance.png -> clean F1 at the common recall=0.95 operating
                         point, per model, sorted ascending.

Run: PYTHONPATH=. python experiments/16_revision_figures.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from config.paths import TABLES_DIR  # noqa: E402

FIG_DIR = Path(r"D:\BEIHANG UNIVERSITY\Research\papers\_archive\GPS_Solutions_sn\figures")
C_CLASSICAL = '#2c7fb8'
C_DEEP = '#d95f0e'
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})


def fig_decision_asr():
    d = pd.read_csv(TABLES_DIR / 'blackbox_boundary_all.csv')
    d = d[np.isclose(d.epsilon, 0.10)].copy()
    d = d.sort_values(['family', 'asr'], ascending=[True, False])
    colors = [C_CLASSICAL if f == 'classical' else C_DEEP for f in d.family]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(d))
    ax.bar(x, d.asr, color=colors, edgecolor='black', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(d.model, rotation=40, ha='right')
    cmean = d[d.family == 'classical'].asr.mean()
    dmean = d[d.family == 'deep'].asr.mean()
    ax.axhline(cmean, color=C_CLASSICAL, ls='--', lw=1.3,
               label=f'classical mean {cmean:.2f}')
    ax.axhline(dmean, color=C_DEEP, ls='--', lw=1.3,
               label=f'deep mean {dmean:.2f}')
    ax.set_ylabel(r'Attack success rate at $\epsilon=0.10$')
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc='upper right')
    ax.set_title('Decision-based attack: no architectural advantage')
    fig.tight_layout()
    out = FIG_DIR / 'fig09_spearman.png'
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out}")


def fig_clean_f1():
    d = pd.read_csv(TABLES_DIR / 'operating_point_recall95.csv')
    d = d.sort_values('f1')
    colors = [C_CLASSICAL if f == 'classical' else C_DEEP for f in d.family]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(d))
    ax.bar(x, d.f1, color=colors, edgecolor='black', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(d.model, rotation=40, ha='right')
    ax.set_ylim(0.80, 0.95)
    ax.set_ylabel(r'Clean $F_1$ at recall$=0.95$ operating point')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_CLASSICAL, label='classical'),
                       Patch(color=C_DEEP, label='deep')],
              frameon=False, loc='lower right')
    ax.set_title('Clean performance at the common operating point')
    fig.tight_layout()
    out = FIG_DIR / 'fig01_clean_performance.png'
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out}")


if __name__ == '__main__':
    fig_decision_asr()
    fig_clean_f1()
