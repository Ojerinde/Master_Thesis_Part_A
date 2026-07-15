"""
Regenerate the data-driven manuscript figures from the single-run tables, so
figures and tables agree. Writes into the manuscript figures dir (stable names).
  fig08_worst_case_recall  fig05_pgd_monotonicity  fig03_adv_heatmap
  fig15_asr_grouped        fig12_radar_summary
Run: PYTHONPATH=. python experiments/21_figures.py
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

FIG = Path(r"D:\BEIHANG UNIVERSITY\Research\manuscript\GPS_Solutions_sn\figures")
CC, CD = '#2c7fb8', '#d95f0e'
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

d = pd.read_csv(TABLES_DIR / 'adversarial_full_oppoint.csv')
bb = pd.read_csv(TABLES_DIR / 'blackbox_boundary_all.csv')
clean = d[d.attack == 'clean'].set_index('model')
DLM = ['CNN-1D', 'LSTM', 'BiLSTM', 'CNN-LSTM', 'Transformer', 'TCN']
CLM = ['RandomForest', 'XGBoost', 'LightGBM', 'GradientBoosting', 'KNN', 'MLP', 'DecisionTree']


def worst_recall():
    bb20 = bb[np.isclose(bb.epsilon, 0.20)].set_index('model').asr
    out = {}
    for m in DLM + CLM:
        sub = d[(d.model == m) & (d.attack != 'clean')]
        dec = clean.loc[m, 'recall'] * (1 - bb20.get(m, 0))
        out[m] = min(sub.recall.min(), dec)
    return out


def fig08():
    wr = worst_recall()
    order = DLM + CLM
    x = np.arange(len(order)); w = 0.4
    cols = [CD if m in DLM else CC for m in order]
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.bar(x - w/2, [clean.loc[m, 'recall'] for m in order], w, label='clean recall', color='0.7', edgecolor='k', linewidth=.3)
    ax.bar(x + w/2, [wr[m] for m in order], w, label='worst-case recall', color=cols, edgecolor='k', linewidth=.3)
    ax.axhline(0.95, ls=':', c='k', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=40, ha='right')
    ax.set_ylabel('Recall'); ax.set_ylim(0, 1.02)
    ax.set_title('Worst-case recall collapses for both families (decision-based)')
    ax.legend(frameon=False, loc='upper right')
    fig.tight_layout(); fig.savefig(FIG / 'fig08_worst_case_recall.png'); plt.close(fig)
    print('fig08')


def fig05():
    eps = [0.05, 0.10, 0.20]
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for m in DLM:
        r = [d[(d.model == m) & (d.attack == 'PGD') & (np.isclose(d.eps, e))].recall.iloc[0] for e in eps]
        lw, mk = (2.6, 'o') if m == 'Transformer' else (1.3, '.')
        ax.plot(eps, r, marker=mk, lw=lw, label=m)
    ax.set_xlabel(r'$\varepsilon$'); ax.set_ylabel('PGD adversarial recall')
    ax.set_title('PGD recall vs. budget (Transformer is non-monotonic)')
    ax.legend(frameon=False, fontsize=9); ax.set_xticks(eps)
    fig.tight_layout(); fig.savefig(FIG / 'fig05_pgd_monotonicity.png'); plt.close(fig)
    print('fig05')


def fig03():
    atks = ['FGSM', 'PGD', 'PGD-Transfer', 'PGD-Transfer-Multi', 'DLSA', 'SNA', 'TPA']
    order = DLM + CLM
    M = np.full((len(order), len(atks)), np.nan)
    for i, m in enumerate(order):
        for j, a in enumerate(atks):
            s = d[(d.model == m) & (d.attack == a) & (np.isclose(d.eps, 0.10))]
            if len(s):
                M[i, j] = clean.loc[m, 'f1'] - s.f1.iloc[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(M, aspect='auto', cmap='magma_r', vmin=0, vmax=0.35)
    ax.set_xticks(range(len(atks))); ax.set_xticklabels(atks, rotation=35, ha='right')
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    ax.axhline(len(DLM) - 0.5, color='k', lw=1.5, ls='--')
    for i in range(len(order)):
        for j in range(len(atks)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f'{M[i,j]:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if M[i, j] > 0.18 else 'black')
    fig.colorbar(im, ax=ax, label=r'$\Delta F_1$ at $\varepsilon=0.10$')
    ax.set_title(r'$\Delta F_1$ by model and attack (DL top, classical bottom)')
    fig.tight_layout(); fig.savefig(FIG / 'fig03_adv_heatmap.png'); plt.close(fig)
    print('fig03')


def fig15():
    # mean decision-based ASR by family across eps
    fig, ax = plt.subplots(figsize=(6.5, 4))
    eps = [0.05, 0.10, 0.20]; x = np.arange(len(eps)); w = 0.38
    for k, (fam, lab, c) in enumerate([('deep', 'DL', CD), ('classical', 'Classical', CC)]):
        means = [bb[(bb.family == fam) & (np.isclose(bb.epsilon, e))].asr.mean() for e in eps]
        ax.bar(x + (k - 0.5) * w, means, w, label=lab, color=c, edgecolor='k', linewidth=.3)
    ax.set_xticks(x); ax.set_xticklabels([f'{e}' for e in eps])
    ax.set_xlabel(r'$\varepsilon$'); ax.set_ylabel('Mean decision-based ASR')
    ax.set_ylim(0, 1); ax.legend(frameon=False)
    ax.set_title('Decision-based attack success: DL vs classical (near-identical)')
    fig.tight_layout(); fig.savefig(FIG / 'fig15_asr_grouped.png'); plt.close(fig)
    print('fig15')


def fig12():
    wr = worst_recall()
    bb10 = bb[np.isclose(bb.epsilon, 0.10)].set_index('model').asr
    metrics = ['Clean recall', 'Clean F1', 'Worst-case\nrecall', 'Decision\nASR@0.10']
    fam_vals = {}
    for fam, ms in [('DL', DLM), ('Classical', CLM)]:
        fam_vals[fam] = [np.mean([clean.loc[m, 'recall'] for m in ms]),
                         np.mean([clean.loc[m, 'f1'] for m in ms]),
                         np.mean([wr[m] for m in ms]),
                         np.mean([bb10[m] for m in ms])]
    x = np.arange(len(metrics)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - w/2, fam_vals['DL'], w, label='DL', color=CD, edgecolor='k', linewidth=.3)
    ax.bar(x + w/2, fam_vals['Classical'], w, label='Classical', color=CC, edgecolor='k', linewidth=.3)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1); ax.set_ylabel('Mean value'); ax.legend(frameon=False)
    ax.set_title('DL vs classical across clean and adversarial axes')
    fig.tight_layout(); fig.savefig(FIG / 'fig12_radar_summary.png'); plt.close(fig)
    print('fig12')


def _dfm(model, attack, eps):
    s = d[(d.model == model) & (d.attack == attack) & (np.isclose(d.eps, eps))]
    return (clean.loc[model, 'f1'] - s.f1.iloc[0]) if len(s) else np.nan


def fig06():
    atks = ['PGD-Transfer', 'PGD-Transfer-Multi', 'DLSA', 'SNA', 'TPA']
    labs = ['PGD-T', 'PGD-T-Multi', 'DLSA', 'SNA', 'TPA']
    x = np.arange(len(CLM)); w = 0.16
    fig, ax = plt.subplots(figsize=(10, 4.3))
    for j, (a, lb) in enumerate(zip(atks, labs)):
        ax.bar(x + (j - 2) * w, [_dfm(m, a, 0.10) for m in CLM], w, label=lb, edgecolor='k', linewidth=.2)
    ax.set_xticks(x); ax.set_xticklabels(CLM, rotation=40, ha='right')
    ax.set_ylabel(r'$\Delta F_1$ at $\varepsilon=0.10$'); ax.legend(frameon=False, fontsize=8, ncol=5)
    ax.set_title('Classical models: transfer beats DLSA/SNA (but decision-based dominates)')
    fig.tight_layout(); fig.savefig(FIG / 'fig06_transfer_classical.png'); plt.close(fig); print('fig06')


def fig07():
    eps = [0.05, 0.10, 0.20]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (fam, ms, ttl) in zip(axes, [('deep', DLM, 'DL'), ('classical', CLM, 'Classical')]):
        for a, c in [('DLSA', CD), ('SNA', CC), ('TPA', '0.4')]:
            means = [np.nanmean([_dfm(m, a, e) for m in ms]) for e in eps]
            ax.plot(eps, means, marker='o', label=a, color=c)
        ax.set_title(ttl); ax.set_xlabel(r'$\varepsilon$'); ax.set_xticks(eps); ax.legend(frameon=False)
    axes[0].set_ylabel(r'mean $\Delta F_1$')
    fig.suptitle('Domain-specific attacks affect DL more than classical')
    fig.tight_layout(); fig.savefig(FIG / 'fig07_domain_attacks.png'); plt.close(fig); print('fig07')


def fig13():
    eps = [0.05, 0.10, 0.20]; atks = ['FGSM', 'PGD', 'DLSA', 'SNA', 'TPA']
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2), sharey=True)
    for ax, a in zip(axes, atks):
        for m in DLM:
            ax.plot(eps, [_dfm(m, a, e) for e in eps], marker='.', label=m)
        ax.set_title(a); ax.set_xlabel(r'$\varepsilon$'); ax.set_xticks(eps)
    axes[0].set_ylabel(r'$\Delta F_1$'); axes[-1].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / 'fig13_delta_f1_by_attack.png'); plt.close(fig); print('fig13')


if __name__ == '__main__':
    fig08(); fig05(); fig03(); fig15(); fig12(); fig06(); fig07(); fig13()
