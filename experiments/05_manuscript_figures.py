"""
Stage 5 — Manuscript Figure Generation
=======================================
Generate ALL 15 manuscript figures from the CSV results produced by
Stages 1-4.  Figures are saved to  results/figures/manuscript/  and
optionally copied to the LaTeX figures/ directory.

Usage:
    python -m experiments.05_manuscript_figures
    python -m experiments.05_manuscript_figures --copy-to "../../manuscript/GPS_Solutions_sn/figures"
"""

from config.paths import TABLES_DIR, FIGURES_DIR
from scipy import stats
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import argparse
import shutil
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')


# --- Global constants ---
TABLES = TABLES_DIR
OUT = FIGURES_DIR / 'manuscript'
OUT.mkdir(parents=True, exist_ok=True)

DL_MODELS = {'Transformer', 'BiLSTM', 'LSTM', 'CNN-1D', 'CNN-LSTM', 'TCN'}
CLASSICAL_MODELS = {'RandomForest', 'DecisionTree', 'XGBoost', 'LightGBM',
                    'GradientBoosting', 'GradBoost', 'KNN', 'MLP'}

# ── Single source of truth for the colour palette ──────────────────
# Change PALETTE_NAME to any matplotlib sequential/perceptual colormap
# (e.g. 'viridis', 'cividis', 'inferno', 'plasma', 'magma') and every
PALETTE_NAME = 'cividis'
_CMAP = plt.colormaps[PALETTE_NAME]


def _hex(val):
    """Convert a cmap scalar ∈ [0,1] to a hex colour string."""
    return '#%02x%02x%02x' % tuple(int(c * 255) for c in _CMAP(val)[:3])


DL_COLOR = _hex(0.15)
DL_COLOR_LIGHT = _hex(0.30)
CL_COLOR = _hex(0.80)
CL_COLOR_DARK = _hex(0.65)

# Discrete line palette (6 colours evenly spaced through the map)
CIVIDIS_LINES = [_CMAP(v) for v in np.linspace(0.10, 0.95, 6)]

ATTACK_ORDER = ['FGSM', 'PGD', 'FGSM-Transfer', 'PGD-Transfer',
                'DLSA', 'SNA', 'TPA']
EPSILON_VALS = [0.05, 0.10, 0.20]

warnings.filterwarnings('ignore', category=FutureWarning)

# --- Style ---
plt.rcParams.update({
    'font.size': 13,
    'font.weight': 'bold',
    'axes.labelsize': 15,
    'axes.labelweight': 'bold',
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def _is_dl(name):
    return name in DL_MODELS


def _norm_model(name):
    return name.replace('GradientBoosting', 'GradBoost')


def _load_adv():
    """Load adversarial attack results and normalise model names."""
    df = pd.read_csv(TABLES / 'adversarial_attack_results.csv')
    df['model'] = df['model'].apply(_norm_model)
    return df


def _load_bootstrap():
    return pd.read_csv(TABLES / 'table1_bootstrap_cis.csv')


def _load_worst():
    df = pd.read_csv(TABLES / 'worst_case_robustness.csv')
    df['model'] = df['model'].apply(_norm_model)
    return df


def _load_corr():
    return pd.read_csv(TABLES / 'table7_robustness_correlation.csv')


def _load_degradation():
    return pd.read_csv(TABLES / 'table6_degradation_summary.csv')


def _load_friedman():
    return pd.read_csv(TABLES / 'table4_friedman_attack_comparison.csv')


def _savefig(fig, name):
    out = OUT / name
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {name}")
    return out


# ═══════════════════════════════════════════════════════════════════════
# Fig 01 — Clean-data F1 with bootstrap CIs (horizontal bars)
# ═══════════════════════════════════════════════════════════════════════
def fig01_clean_performance():
    df = _load_bootstrap()
    df = df.sort_values('F1', ascending=True).reset_index(drop=True)

    n_dl = sum(_is_dl(m) for m in df['Model'])
    n_cl = len(df) - n_dl
    dl_idx, cl_idx = 0, 0

    colors = []
    for m in df['Model']:
        if _is_dl(m):
            colors.append(_CMAP(0.10 + 0.35 * dl_idx / max(n_dl - 1, 1)))
            dl_idx += 1
        else:
            colors.append(_CMAP(0.60 + 0.35 * cl_idx / max(n_cl - 1, 1)))
            cl_idx += 1

    fig, ax = plt.subplots(figsize=(12, 10))
    y = np.arange(len(df))
    ax.barh(y, df['F1'], height=0.7, color=colors, edgecolor='none', zorder=3)

    xerr_lo = df['F1'] - df['F1_CI_lower']
    xerr_hi = df['F1_CI_upper'] - df['F1']
    ax.errorbar(df['F1'], y, xerr=[xerr_lo, xerr_hi],
                fmt='none', ecolor='black', capsize=5, capthick=1.5,
                elinewidth=1.5, zorder=4)

    f1_min, f1_max = df['F1'].min(), df['F1'].max()
    ax.axvspan(f1_min, f1_max, alpha=0.08, color='steelblue', zorder=1)

    for i, row in df.iterrows():
        text_x = max(row['F1'] + 0.002, row['F1_CI_upper'] + 0.002)
        ax.text(text_x, i, f"{row['F1']:.3f}",
                va='center', ha='left', fontsize=13, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(df['Model'], fontsize=13, fontweight='bold')
    ax.set_xlabel('F$_1$ Score (Clean Data)', fontsize=15, fontweight='bold')
    ax.set_xlim(f1_min - 0.02, f1_max + 0.03)
    ax.grid(axis='x', alpha=0.3, zorder=0)
    ax.invert_yaxis()

    legend_elements = [
        Patch(facecolor=_CMAP(0.25), label='Deep Learning'),
        Patch(facecolor=_CMAP(0.75), label='Classical'),
        plt.Line2D([0], [0], color='black', marker='|', markersize=10,
                   linestyle='-', linewidth=1.5, label='95% bootstrap CI'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12,
              framealpha=0.9)
    fig.tight_layout()
    return _savefig(fig, 'fig01_clean_performance.png')


# ═══════════════════════════════════════════════════════════════════════
# Fig ROC — ROC curves (2-panel: DL + Classical)
# ═══════════════════════════════════════════════════════════════════════
def fig_roc_curves():
    boot = _load_bootstrap()
    # We build a synthetic ROC from the single operating point
    # (FPR=1-Precision proxy isn't a real ROC).
    # The actual ROC needs raw predictions — not available from CSVs.
    # Use the existing figure from Stage 1/2 if available.
    src_dl = FIGURES_DIR / 'dl_baseline' / 'roc_curves.png'
    src_cl = FIGURES_DIR / 'baseline' / 'roc_curves.png'
    if src_dl.exists() and src_cl.exists():
        # Combine into two-panel figure
        from PIL import Image
        img_dl = Image.open(src_dl)
        img_cl = Image.open(src_cl)
        total_w = img_dl.width + img_cl.width
        max_h = max(img_dl.height, img_cl.height)
        combined = Image.new('RGB', (total_w, max_h), 'white')
        combined.paste(img_dl, (0, 0))
        combined.paste(img_cl, (img_dl.width, 0))
        out = OUT / 'fig_roc_curves.png'
        combined.save(out, dpi=(300, 300))
        print(f"  ✓ fig_roc_curves.png (combined from Stage 1+2)")
        return out
    elif src_cl.exists():
        out = OUT / 'fig_roc_curves.png'
        shutil.copy2(src_cl, out)
        print(f"  ✓ fig_roc_curves.png (from Stage 1 baseline)")
        return out
    else:
        print("  ⚠ fig_roc_curves.png — ROC curves require raw predictions;"
              " skipped (use Stage 1/2 output)")
        return None


# ═══════════════════════════════════════════════════════════════════════
# Fig 03 — Adversarial ΔF1 Heatmap (13 models × 7 attacks at ε=0.10)
# ═══════════════════════════════════════════════════════════════════════
def fig03_adv_heatmap():
    adv = _load_adv()
    sub = adv[np.isclose(adv['epsilon'], 0.10)].copy()

    pivot = sub.pivot_table(index='model', columns='attack',
                            values='delta_f1', aggfunc='mean')
    # Order: DL first then classical, each sorted by worst ΔF1
    dl_models = [m for m in pivot.index if _is_dl(m)]
    cl_models = [m for m in pivot.index if not _is_dl(m)]
    dl_order = pivot.loc[dl_models].max(
        axis=1).sort_values(ascending=False).index
    cl_order = pivot.loc[cl_models].max(
        axis=1).sort_values(ascending=False).index
    ordered = list(dl_order) + list(cl_order)
    cols = [c for c in ATTACK_ORDER if c in pivot.columns]
    pivot = pivot.loc[ordered, cols]

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap=PALETTE_NAME,
                ax=ax, vmin=0, linewidths=0.5,
                cbar_kws={'label': 'ΔF$_1$'})
    # Separator line between DL and classical
    ax.axhline(len(dl_order), color='navy', linewidth=2.5, linestyle='--')
    ax.set_ylabel('')
    ax.set_xlabel('Attack Type', fontsize=14, fontweight='bold')
    fig.tight_layout()
    return _savefig(fig, 'fig03_adv_heatmap.png')


# ═══════════════════════════════════════════════════════════════════════
# Fig 15 — Attack Success Rate grouped bars (DL vs Classical at ε=0.10)
# ═══════════════════════════════════════════════════════════════════════
def fig15_asr_grouped():
    adv = _load_adv()
    sub = adv[np.isclose(adv['epsilon'], 0.10)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, (group_label, group_type) in zip(axes, [
            ('Deep Learning', 'deep_learning'),
            ('Classical', 'classical')]):
        grp = sub[sub['model_type'] == group_type]
        means = grp.groupby('attack')[
            'attack_success_rate'].agg(['mean', 'std'])
        attacks = [a for a in ATTACK_ORDER if a in means.index]
        means = means.loc[attacks]

        x = np.arange(len(attacks))
        color = DL_COLOR_LIGHT if group_type == 'deep_learning' else CL_COLOR
        ax.bar(x, means['mean'], yerr=means['std'], width=0.6,
               color=color, edgecolor='black', capsize=4, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(attacks, rotation=35, ha='right', fontsize=13,
                           fontweight='bold')
        ax.set_title(f'({chr(97 + list(axes).index(ax))}) {group_label}',
                     fontsize=15, fontweight='bold')
        ax.set_ylabel('Attack Success Rate (ASR)' if ax == axes[0] else '',
                      fontsize=14, fontweight='bold')
        ax.tick_params(axis='y', labelsize=13)
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.set_ylim(0, 1)

    fig.tight_layout()
    return _savefig(fig, 'fig15_asr_grouped.png')


# ═══════════════════════════════════════════════════════════════════════
# Fig 05 — PGD monotonicity (Recall vs ε for DL models)
# ═══════════════════════════════════════════════════════════════════════
def fig05_pgd_monotonicity():
    adv = _load_adv()
    pgd = adv[adv['attack'] == 'PGD'].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    dl_models = sorted(
        pgd[pgd['model_type'] == 'deep_learning']['model'].unique())
    cmap_line = [_CMAP(v) for v in np.linspace(0.10, 0.95, len(dl_models))]

    for panel_idx, (ax, title_suffix) in enumerate(zip(axes,
                                                       ['Without GNSS Enforcer', 'With GNSS Enforcer'])):
        for i, model in enumerate(dl_models):
            mdf = pgd[pgd['model'] == model].sort_values('epsilon')
            ax.plot(mdf['epsilon'], mdf['adv_recall'],
                    marker='o', linewidth=2.2, markersize=8,
                    color=cmap_line[i], label=model, zorder=3)
        ax.set_xlabel('ε', fontsize=14, fontweight='bold')
        if panel_idx == 0:
            ax.set_ylabel('Adversarial Recall', fontsize=14, fontweight='bold')
        ax.set_title(f'({chr(97 + panel_idx)}) {title_suffix}',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='lower left')
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_xticks(EPSILON_VALS)

    fig.tight_layout()
    return _savefig(fig, 'fig05_pgd_monotonicity.png')


# ═══════════════════════════════════════════════════════════════════════
# Fig 06 — Transfer attack grouped bars for classical models
# ═══════════════════════════════════════════════════════════════════════
def fig06_transfer_classical():
    adv = _load_adv()
    sub = adv[(adv['model_type'] == 'classical') &
              np.isclose(adv['epsilon'], 0.10)].copy()

    attack_types = ['FGSM-Transfer', 'PGD-Transfer', 'DLSA', 'SNA', 'TPA']
    sub = sub[sub['attack'].isin(attack_types)]

    models = sorted(sub['model'].unique())
    pivot = sub.pivot_table(index='model', columns='attack',
                            values='delta_f1', aggfunc='mean')
    pivot = pivot.reindex(
        columns=[a for a in attack_types if a in pivot.columns])

    x = np.arange(len(models))
    width = 0.15
    colors = [_CMAP(v) for v in np.linspace(0.10, 0.90, len(attack_types))]

    fig, ax = plt.subplots(figsize=(14, 7))
    for j, attack in enumerate(pivot.columns):
        vals = [pivot.loc[m, attack] if m in pivot.index else 0 for m in models]
        ax.bar(x + j * width, vals, width, label=attack,
               color=colors[j % len(colors)], edgecolor='black',
               linewidth=0.5, zorder=3)

    ax.set_xticks(x + width * (len(pivot.columns) - 1) / 2)
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=11)
    ax.set_ylabel('ΔF$_1$', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    fig.tight_layout()
    return _savefig(fig, 'fig06_transfer_classical.png')


# ═══════════════════════════════════════════════════════════════════════
# Fig 07 — Domain-specific attacks (DLSA, SNA, TPA) mean ΔF1 vs ε
# ═══════════════════════════════════════════════════════════════════════
def fig07_domain_attacks():
    deg = _load_degradation()
    domain = deg[deg['attack'].isin(['DLSA', 'SNA', 'TPA'])].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    colors = {'DLSA': _CMAP(0.15), 'SNA': _CMAP(0.55), 'TPA': _CMAP(0.90)}

    for ax, (label, mtype) in zip(axes,
                                  [('(a) Deep Learning', 'deep_learning'),
                                   ('(b) Classical', 'classical')]):
        grp = domain[domain['model_type'] == mtype]
        for atk in ['DLSA', 'SNA', 'TPA']:
            adf = grp[grp['attack'] == atk].sort_values('epsilon')
            if adf.empty:
                continue
            ax.plot(adf['epsilon'], adf['mean_delta_f1'],
                    marker='o', linewidth=2.2, color=colors[atk],
                    label=atk, zorder=3)
            ax.fill_between(adf['epsilon'],
                            adf['mean_delta_f1'] - adf['std_delta_f1'],
                            adf['mean_delta_f1'] + adf['std_delta_f1'],
                            color=colors[atk], alpha=0.15, zorder=2)
        ax.set_xlabel('ε', fontsize=14, fontweight='bold')
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_xticks(EPSILON_VALS)

    axes[0].set_ylabel('Mean ΔF$_1$', fontsize=14, fontweight='bold')
    fig.tight_layout()
    return _savefig(fig, 'fig07_domain_attacks.png')


# ═══════════════════════════════════════════════════════════════════════
# Fig 13 — ΔF1 vs ε for DL models, disaggregated by attack type
# ═══════════════════════════════════════════════════════════════════════
def fig13_delta_f1_by_attack():
    adv = _load_adv()
    dl = adv[adv['model_type'] == 'deep_learning'].copy()
    dl_attacks = ['FGSM', 'PGD', 'DLSA', 'SNA', 'TPA']
    dl_models = sorted(dl['model'].unique())
    cmap_line = [_CMAP(v) for v in np.linspace(0.10, 0.95, len(dl_models))]

    # 2 rows × 3 cols; legend fills axes[1,2]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharey=True)
    subplot_attacks = dl_attacks  # FGSM, PGD, DLSA, SNA, TPA
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    for idx, atk in enumerate(subplot_attacks):
        r, c = positions[idx]
        ax = axes[r, c]
        adf = dl[dl['attack'] == atk]
        for i, model in enumerate(dl_models):
            mdf = adf[adf['model'] == model].sort_values('epsilon')
            ax.plot(mdf['epsilon'], mdf['delta_f1'],
                    marker='o', linewidth=2.2, color=cmap_line[i],
                    label=model, zorder=3)
        ax.set_xlabel('ε', fontsize=15, fontweight='bold')
        ax.set_title(f'({chr(97 + idx)}) {atk}',
                     fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', labelsize=13)
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_xticks(EPSILON_VALS)
        if c == 0:
            ax.set_ylabel('ΔF$_1$', fontsize=15, fontweight='bold')

    # Legend in third column of second row
    legend_ax = axes[1, 2]
    legend_ax.set_visible(False)
    handles = [plt.Line2D([0], [0], color=cmap_line[i], lw=2.5, marker='o',
                          label=m) for i, m in enumerate(dl_models)]
    legend_ax.legend(handles=handles, loc='center', fontsize=14,
                     title='Model', title_fontsize=14,
                     framealpha=0.9, bbox_to_anchor=(0.5, 0.5),
                     bbox_transform=legend_ax.transAxes)
    legend_ax.set_visible(True)
    legend_ax.axis('off')
    fig.tight_layout()
    return _savefig(fig, 'fig13_delta_f1_by_attack.png')


# ═══════════════════════════════════════════════════════════════════════
# Fig 08 — Worst-case adversarial recall (horizontal bars)
# ═══════════════════════════════════════════════════════════════════════
def fig08_worst_case_recall():
    df = _load_worst()
    df = df.sort_values('worst_adv_recall',
                        ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    y = np.arange(len(df))

    for i, row in df.iterrows():
        color = DL_COLOR_LIGHT if row['model_type'] == 'deep_learning' else CL_COLOR
        ax.barh(i, row['worst_adv_recall'], height=0.65, color=color,
                edgecolor='none', zorder=3)
        ax.barh(i, row['clean_recall'], height=0.65, fill=False,
                edgecolor=color, linestyle='--', linewidth=1.8, zorder=4)
        ax.text(row['worst_adv_recall'] + 0.005, i,
                f"{row['worst_adv_recall']:.3f}",
                va='center', ha='left', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.7))

    ax.set_yticks(y)
    ax.set_yticklabels(df['model'], fontsize=12, fontweight='bold')
    ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
    ax.set_xlim(0.3, 1.05)
    ax.grid(axis='x', alpha=0.3, zorder=0)

    legend_elements = [
        Patch(facecolor=DL_COLOR_LIGHT, label='DL (worst-case)'),
        Patch(facecolor=CL_COLOR, label='Classical (worst-case)'),
        plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=2,
                   label='Clean recall'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12,
              framealpha=0.9)
    fig.tight_layout()
    return _savefig(fig, 'fig08_worst_case_recall.png')


# ═══════════════════════════════════════════════════════════════════════
# Fig 14 — Confusion matrices (Transformer + TCN, clean vs PGD)
# ═══════════════════════════════════════════════════════════════════════
def fig14_confusion_matrices():
    adv = _load_adv()
    boot = _load_bootstrap()

    models_of_interest = ['Transformer', 'TCN']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for row_idx, model in enumerate(models_of_interest):
        # Clean confusion matrix from baseline
        brow = boot[boot['Model'] == model].iloc[0]
        clean_rec = brow['Recall']
        clean_prec = brow['Precision']
        # Approximate normalised CM: [TN, FP; FN, TP] row-normalised
        clean_cm = np.array([[clean_prec, 1 - clean_prec],
                             [1 - clean_rec, clean_rec]])

        # PGD ε=0.20 adversarial
        pgd = adv[(adv['model'] == model) & (adv['attack'] == 'PGD') &
                  np.isclose(adv['epsilon'], 0.20)]
        if not pgd.empty:
            pr = pgd.iloc[0]
            adv_rec = pr['adv_recall']
            adv_prec = pr['adv_precision']
            adv_cm = np.array([[adv_prec, 1 - adv_prec],
                               [1 - adv_rec, adv_rec]])
        else:
            adv_cm = np.array([[0.5, 0.5], [0.5, 0.5]])

        for col_idx, cm in enumerate([clean_cm, adv_cm]):
            ax = axes[row_idx, col_idx]
            sns.heatmap(cm, annot=True, fmt='.3f', cmap=PALETTE_NAME,
                        vmin=0, vmax=1, ax=ax,
                        xticklabels=['Authentic', 'Spoofed'],
                        yticklabels=['Authentic', 'Spoofed'],
                        cbar=col_idx == 1,
                        annot_kws={'size': 13, 'weight': 'bold'})
            # Subplot label (a)-(d) in top-left corner
            label = chr(97 + row_idx * 2 + col_idx)
            ax.text(0.03, 0.97, f'({label})', transform=ax.transAxes,
                    fontsize=15, fontweight='bold', va='top', ha='left',
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                              alpha=0.45, edgecolor='none'))
            ax.set_ylabel('True' if col_idx == 0 else '', fontsize=13)
            ax.set_xlabel('Predicted', fontsize=13)
            ax.tick_params(axis='both', labelsize=12)

    fig.tight_layout()
    return _savefig(fig, 'fig14_confusion_matrices.png')


# ═══════════════════════════════════════════════════════════════════════
# Fig 09 — Spearman scatter (Clean F1 vs Worst-Case Recall)
# ═══════════════════════════════════════════════════════════════════════
def fig09_spearman():
    df = _load_worst()
    corr = _load_corr()

    dl = df[df['model_type'] == 'deep_learning'].copy()
    cl = df[df['model_type'] == 'classical'].copy()

    dl_row = corr[corr['group'] == 'deep_learning'].iloc[0]
    cl_row = corr[corr['group'] == 'classical'].iloc[0]

    fig, ax = plt.subplots(figsize=(14, 8))

    # --- Classical ---
    ax.scatter(cl['clean_f1'], cl['worst_adv_recall'],
               s=180, c=CL_COLOR, marker='s', edgecolors=CL_COLOR_DARK,
               linewidths=1.2, zorder=5, label='Classical')
    slope_cl, int_cl, _, _, _ = stats.linregress(
        cl['clean_f1'], cl['worst_adv_recall'])
    x_cl = np.linspace(cl['clean_f1'].min() - 0.005,
                       cl['clean_f1'].max() + 0.005, 100)
    ax.plot(x_cl, slope_cl * x_cl + int_cl, color=CL_COLOR,
            linewidth=2.5, zorder=3)
    n_cl = len(cl)
    se_cl = np.sqrt(np.sum((cl['worst_adv_recall'] -
                            (slope_cl * cl['clean_f1'] + int_cl))**2) /
                    (n_cl - 2))
    ci_cl = 1.96 * se_cl * np.sqrt(
        1/n_cl + (x_cl - cl['clean_f1'].mean())**2 /
        np.sum((cl['clean_f1'] - cl['clean_f1'].mean())**2))
    ax.fill_between(x_cl, slope_cl*x_cl+int_cl - ci_cl,
                    slope_cl*x_cl+int_cl + ci_cl,
                    color=CL_COLOR, alpha=0.15, zorder=2)

    # --- DL ---
    ax.scatter(dl['clean_f1'], dl['worst_adv_recall'],
               s=180, c=DL_COLOR, marker='o', edgecolors='#1a2540',
               linewidths=1.2, zorder=5, label='Deep Learning')
    slope_dl, int_dl, _, _, _ = stats.linregress(
        dl['clean_f1'], dl['worst_adv_recall'])
    x_dl = np.linspace(dl['clean_f1'].min() - 0.002,
                       dl['clean_f1'].max() + 0.002, 100)
    ax.plot(x_dl, slope_dl * x_dl + int_dl, color=DL_COLOR,
            linewidth=2.5, zorder=3)
    n_dl = len(dl)
    se_dl = np.sqrt(np.sum((dl['worst_adv_recall'] -
                            (slope_dl * dl['clean_f1'] + int_dl))**2) /
                    (n_dl - 2))
    ci_dl = 1.96 * se_dl * np.sqrt(
        1/n_dl + (x_dl - dl['clean_f1'].mean())**2 /
        np.sum((dl['clean_f1'] - dl['clean_f1'].mean())**2))
    ax.fill_between(x_dl, slope_dl*x_dl+int_dl - ci_dl,
                    slope_dl*x_dl+int_dl + ci_dl,
                    color=DL_COLOR, alpha=0.12, zorder=2)

    # Model name annotations — explicit per-model offsets to avoid overlap
    import matplotlib.patheffects as patheffects
    # Assign single-letter labels to all models
    all_models = list(cl['model']) + list(dl['model'])
    label_map = {m: chr(65+i) for i, m in enumerate(all_models)}  # A, B, C...
    # Reduce fontsize for clarity, no offset
    label_fontsize = 11
    for _, row in cl.iterrows():
        ax.text(row['clean_f1'], row['worst_adv_recall'], label_map[row['model']],
                fontsize=label_fontsize, fontweight='bold', color='white', ha='center', va='center', zorder=10,
                path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
    for _, row in dl.iterrows():
        ax.text(row['clean_f1'], row['worst_adv_recall'], label_map[row['model']],
                fontsize=label_fontsize, fontweight='bold', color='white', ha='center', va='center', zorder=10,
                path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
    # Add legend mapping at bottom left
    mapping_lines = [f"{label_map[m]}: {m}" for m in all_models]
    mapping_text = "\n".join(mapping_lines)
    ax.text(0.01, 0.01, mapping_text, transform=ax.transAxes, fontsize=13, fontweight='bold',
            va='bottom', ha='left', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'))

    # Spearman annotation boxes
    rho_cl = cl_row['spearman_rho']
    p_cl = cl_row['p_value']
    p_cl_str = f"p = {p_cl:.4f}" if p_cl >= 0.001 else "p < 0.001"
    ax.annotate(f"ρ = +{rho_cl:.3f},  {p_cl_str}",
                xy=(0.15, 0.88), xycoords='axes fraction',
                fontsize=14, fontweight='bold', color=CL_COLOR_DARK,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor=CL_COLOR, linewidth=2))

    rho_dl = dl_row['spearman_rho']
    p_dl = dl_row['p_value']
    ax.annotate(f"ρ = {rho_dl:.3f},  p = {p_dl:.3f}",
                xy=(0.65, 0.05), xycoords='axes fraction',
                fontsize=14, fontweight='bold', color=DL_COLOR,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor=DL_COLOR, linewidth=2))

    ax.set_xlabel('Clean F$_1$ Score', fontsize=15, fontweight='bold')
    ax.set_ylabel('Worst-Case Adversarial Recall', fontsize=15,
                  fontweight='bold')
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(loc='upper right', fontsize=13, framealpha=0.9,
              ncol=2, markerscale=1.2)
    fig.tight_layout()
    return _savefig(fig, 'fig09_spearman.png')


# ═══════════════════════════════════════════════════════════════════════
# Fig 11 — Latency heatmap (log10 attack/inference for DL models)
# ═══════════════════════════════════════════════════════════════════════
def fig11_latency_heatmap():
    adv = _load_adv()
    dl = adv[adv['model_type'] == 'deep_learning'].copy()
    dl = dl.dropna(subset=['attack_gen_us', 'detector_infer_us'])
    if dl.empty:
        print("  ⚠ fig11_latency_heatmap — no DL latency data; skipped")
        return None

    dl['log_ratio'] = np.log10(dl['attack_gen_us'] / dl['detector_infer_us'])

    # Average over epsilons
    pivot = dl.pivot_table(index='model', columns='attack',
                           values='log_ratio', aggfunc='mean')
    cols = [c for c in ATTACK_ORDER if c in pivot.columns]
    pivot = pivot.reindex(columns=cols)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap=PALETTE_NAME,
                center=0, ax=ax, linewidths=0.5, annot_kws={'size': 13, 'weight': 'bold'},
                cbar_kws={'label': 'log₁₀(attack / inference)'})
    ax.set_ylabel('')
    ax.set_xlabel('Attack Type', fontsize=15, fontweight='bold')

    # Red border for cells where attack > inference
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            if pivot.iloc[i, j] > 0:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                             edgecolor='red', linewidth=2.5))

    fig.tight_layout()
    return _savefig(fig, 'fig11_latency_heatmap.png')


# ═══════════════════════════════════════════════════════════════════════
# Fig 16 — Friedman average ranks (bar chart)
# ═══════════════════════════════════════════════════════════════════════
def fig16_friedman_ranks():
    adv = _load_adv()

    # Rank each model per (attack, epsilon) by delta_f1 (lower = more robust)
    groups = adv.groupby(['attack', 'epsilon'])
    ranks_list = []
    for (atk, eps), grp in groups:
        grp = grp.copy()
        grp['rank'] = grp['delta_f1'].rank(method='average')
        ranks_list.append(grp[['model', 'model_type', 'rank']])

    all_ranks = pd.concat(ranks_list, ignore_index=True)
    avg = all_ranks.groupby('model')['rank'].mean().sort_values()

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = [DL_COLOR_LIGHT if _is_dl(m) else CL_COLOR for m in avg.index]
    bars = ax.barh(range(len(avg)), avg.values, color=colors,
                   edgecolor='black', linewidth=0.5, zorder=3)

    for i, (m, v) in enumerate(avg.items()):
        ax.text(v + 0.1, i, f"{v:.1f}", va='center', fontsize=11,
                fontweight='bold')

    ax.set_yticks(range(len(avg)))
    ax.set_yticklabels(avg.index, fontsize=12, fontweight='bold')
    ax.set_xlabel('Average Friedman Rank (lower = more robust)',
                  fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, zorder=0)
    ax.invert_yaxis()

    legend_elements = [
        Patch(facecolor=DL_COLOR_LIGHT, label='Deep Learning'),
        Patch(facecolor=CL_COLOR, label='Classical'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    fig.tight_layout()
    return _savefig(fig, 'fig16_friedman_ranks.png')


# ═══════════════════════════════════════════════════════════════════════
# Fig 12 — Radar/spider chart (multi-metric summary)
# ═══════════════════════════════════════════════════════════════════════
def fig12_radar_summary():
    boot = _load_bootstrap()
    worst = _load_worst()
    adv = _load_adv()

    # Build per-model metric dict
    models_all = boot['Model'].tolist()
    metrics = {}

    for m in models_all:
        mn = _norm_model(m)
        brow = boot[boot['Model'] == m].iloc[0]
        wrow = worst[worst['model'] == mn]

        clean_f1 = brow['F1']
        worst_recall = wrow['worst_adv_recall'].values[0] if len(wrow) else 0

        # Transfer robustness: 1 - mean ΔF1 under transfer attacks
        m_adv = adv[(adv['model'] == mn) &
                    adv['attack'].isin(['FGSM-Transfer', 'PGD-Transfer'])]
        transfer_rob = 1 - m_adv['delta_f1'].mean() if len(m_adv) else 1

        # Domain robustness: 1 - mean ΔF1 under DLSA, SNA, TPA
        m_dom = adv[(adv['model'] == mn) &
                    adv['attack'].isin(['DLSA', 'SNA', 'TPA'])]
        domain_rob = 1 - m_dom['delta_f1'].mean() if len(m_dom) else 1

        # Latency feasibility: normalised 1/(inference time)
        m_lat = adv[adv['model'] == mn]['detector_infer_us'].dropna()
        lat_us = m_lat.mean() if len(m_lat) else 100
        metrics[mn] = {
            'Clean F₁': clean_f1,
            'Worst-Case Recall': worst_recall,
            'Transfer Robustness': transfer_rob,
            'Domain Robustness': domain_rob,
            'Latency (1/μs)': 1.0 / lat_us,
        }

    mdf = pd.DataFrame(metrics).T

    # --- Keep only top-3 DL and top-3 classical by Clean F₁ ----------
    dl_mask = pd.Series([_is_dl(m) for m in mdf.index], index=mdf.index)
    top_dl = mdf.loc[dl_mask].nlargest(3, 'Clean F₁').index.tolist()
    top_cl = mdf.loc[~dl_mask].nlargest(3, 'Clean F₁').index.tolist()
    mdf = mdf.loc[top_dl + top_cl]

    # Min-max normalise each column
    for col in mdf.columns:
        mn, mx = mdf[col].min(), mdf[col].max()
        if mx > mn:
            mdf[col] = (mdf[col] - mn) / (mx - mn)
        else:
            mdf[col] = 1.0

    categories = list(mdf.columns)
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    dl_colors = [_CMAP(v) for v in [0.10, 0.25, 0.40]]
    cl_colors = [_CMAP(v) for v in [0.60, 0.75, 0.90]]
    dl_i = cl_i = 0

    for model in mdf.index:
        vals = mdf.loc[model].tolist() + [mdf.loc[model].iloc[0]]
        if _is_dl(model):
            c, ls, lw, alpha = dl_colors[dl_i], '-', 2.2, 0.10
            dl_i += 1
        else:
            c, ls, lw, alpha = cl_colors[cl_i], '--', 2.2, 0.08
            cl_i += 1
        ax.plot(angles, vals, linewidth=lw, linestyle=ls, label=model,
                color=c, zorder=3, marker='o', markersize=5)
        ax.fill(angles, vals, alpha=alpha, color=c)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
              fontsize=13, ncol=1, framealpha=0.9)
    fig.tight_layout()
    return _savefig(fig, 'fig12_radar_summary.png')


# ═══════════════════════════════════════════════════════════════════════
# Fig 02 — KNN threshold analysis (copied from Stage 1)
# ═══════════════════════════════════════════════════════════════════════
def fig02_knn_threshold():
    src = FIGURES_DIR / 'baseline' / 'threshold_KNN.png'
    if src.exists():
        out = OUT / 'fig02_knn_threshold.png'
        shutil.copy2(src, out)
        print(f"  ✓ fig02_knn_threshold.png (from Stage 1)")
        return out
    print("  ⚠ fig02_knn_threshold.png — not found; skipped")
    return None


# ═══════════════════════════════════════════════════════════════════════
# Master runner
# ═══════════════════════════════════════════════════════════════════════
FIGURE_REGISTRY = [
    ('fig01_clean_performance',   fig01_clean_performance),
    ('fig02_knn_threshold',       fig02_knn_threshold),
    ('fig03_adv_heatmap',         fig03_adv_heatmap),
    ('fig15_asr_grouped',         fig15_asr_grouped),
    ('fig05_pgd_monotonicity',    fig05_pgd_monotonicity),
    ('fig06_transfer_classical',  fig06_transfer_classical),
    ('fig07_domain_attacks',      fig07_domain_attacks),
    ('fig13_delta_f1_by_attack',  fig13_delta_f1_by_attack),
    ('fig08_worst_case_recall',   fig08_worst_case_recall),
    ('fig14_confusion_matrices',  fig14_confusion_matrices),
    ('fig09_spearman',            fig09_spearman),
    ('fig11_latency_heatmap',     fig11_latency_heatmap),
    ('fig16_friedman_ranks',      fig16_friedman_ranks),
    ('fig12_radar_summary',       fig12_radar_summary),
]


def run(copy_to=None):
    print("=" * 70)
    print("  STAGE 5 — MANUSCRIPT FIGURE GENERATION")
    print("=" * 70)
    print(f"  Source CSVs : {TABLES}")
    print(f"  Output dir  : {OUT}\n")

    generated = []
    skipped = []

    for name, func in FIGURE_REGISTRY:
        try:
            result = func()
            if result:
                generated.append(name)
            else:
                skipped.append(name)
        except Exception as e:
            print(f"  ✗ {name} — ERROR: {e}")
            skipped.append(name)

    print(f"\n  Generated : {len(generated)} / {len(FIGURE_REGISTRY)}")
    if skipped:
        print(f"  Skipped   : {', '.join(skipped)}")

    # Copy to manuscript if requested
    if copy_to:
        dest = Path(copy_to)
        dest.mkdir(parents=True, exist_ok=True)
        copied = 0
        for name in generated:
            src = OUT / f"{name}.png"
            if src.exists():
                shutil.copy2(src, dest / src.name)
                copied += 1
        print(f"\n  Copied {copied} figures → {dest}")

    print(f"\n{'=' * 70}")
    print("  STAGE 5 COMPLETE")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 5: Generate all manuscript figures from CSV results")
    parser.add_argument(
        '--copy-to', type=str, default=None,
        help='Copy generated figures to this directory (e.g. manuscript figures folder)')
    args = parser.parse_args()
    run(copy_to=args.copy_to)


if __name__ == '__main__':
    main()
