"""
Publication figures for the Satellite Navigation manuscript.
Journal spec: vector PDF (fonts embedded as TrueType/Type 42) + 400 dpi PNG,
sans-serif (Arial/Helvetica, DejaVu Sans fallback), >=7 pt at final size, sized to
the 16 cm text width, colorblind-safe Okabe-Ito palette (validated with the dataviz
palette checker: 2- and 3-way sets PASS CVD separation; orange fills carry dark edges
and direct labels because orange is below the 3:1 surface-contrast floor).

Run: python make_figures.py   (from manuscript/Satellite_Navigation/)
Outputs -> figures/figNN_*.pdf + .png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "code" / "gnss_adversarial_research"
TAB = REPO / "results" / "tables"
FIGD = HERE / "figures"
FIGD.mkdir(exist_ok=True)

# ---- colorblind-safe palette (Okabe-Ito subset, validated) ------------------
BLUE = "#0072B2"     # genuine / classical
VERM = "#D55E00"     # spoof
ORANGE = "#E69F00"   # deep  (low surface contrast -> dark edge + labels on fills)
GREEN = "#009E73"    # 3rd category / accent
INK = "#1a1a1a"
MUTED = "#5c5c5c"
GRID = "#dddddd"
FAM_COLOR = {"classical": BLUE, "deep": ORANGE}
EDGE = {"classical": "#004c78", "deep": "#7a5300"}   # darker rims for legibility

CM = 1 / 2.54
FULL_W = 16 * CM        # journal text width
HALF_W = 8.4 * CM

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.3, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType, no Type-3
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": MUTED,
})


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(FIGD / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  wrote figures/{stem}.pdf + .png")


# ============================================================================
# F6 - Adversarial robustness: decision-based fragility ranking (HEADLINE)
# ============================================================================
def fig_f6():
    bb = pd.read_csv(TAB / "blackbox_boundary_all.csv")
    d = (bb.groupby(["model", "family"], as_index=False)["median_min_linf"].first()
         .sort_values("median_min_linf"))
    ci = pd.read_csv(TAB / "blackbox_boundary_ci.csv").set_index("model")
    d = d.merge(ci[["ci_lo", "ci_hi"]], left_on="model", right_index=True, how="left")
    y = np.arange(len(d))
    colors = [FAM_COLOR[f] for f in d.family]
    edges = [EDGE[f] for f in d.family]

    fig, ax = plt.subplots(figsize=(FULL_W, 8.5 * CM))
    med = d["median_min_linf"].values
    ax.barh(y, med, color=colors, edgecolor=edges, linewidth=0.7,
            height=0.68, zorder=3)
    # 95% bootstrap CI on the median (asymmetric)
    xerr = np.vstack([med - d["ci_lo"].values, d["ci_hi"].values - med])
    ax.errorbar(med, y, xerr=xerr, fmt="none", ecolor=INK, elinewidth=0.8,
                capsize=2, capthick=0.8, zorder=4)
    for yi, hi, v in zip(y, d["ci_hi"].values, med):
        ax.text(hi + 0.002, yi, f"{v:.3f}", va="center", ha="left",
                fontsize=7, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(d["model"])
    ax.set_xlabel(r"Median minimum $\ell_\infty$ perturbation to evade detection"
                  "\n(decision-based attack, 95% bootstrap CI; smaller = more fragile)")
    ax.set_xlim(0, d["ci_hi"].max() * 1.15)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    # family legend (identity not by color alone: also grouped position + this legend)
    handles = [mpl.patches.Patch(facecolor=BLUE, edgecolor=EDGE["classical"],
                                 label="Classical"),
               mpl.patches.Patch(facecolor=ORANGE, edgecolor=EDGE["deep"],
                                 label="Deep")]
    ax.legend(handles=handles, loc="upper right", frameon=False)
    fig.tight_layout()
    save(fig, "fig06_fragility_ranking")


# ============================================================================
# F5 - Clean detection performance at the recall-0.95 operating point
# ============================================================================
def fig_f5():
    op = pd.read_csv(TAB / "operating_point_recall95.csv").sort_values("f1")
    y = np.arange(len(op))
    colors = [FAM_COLOR[f] for f in op.family]
    edges = [EDGE[f] for f in op.family]

    fig, ax = plt.subplots(figsize=(FULL_W, 8.5 * CM))
    ax.barh(y, op["f1"], color=colors, edgecolor=edges, linewidth=0.7,
            height=0.68, zorder=3)
    # label inside the bar; white on the darker blue, dark ink on the lighter orange
    # (orange is below the 3:1 surface-contrast floor, so white would be illegible)
    for yi, v, fam in zip(y, op["f1"], op.family):
        ax.text(v - 0.008, yi, f"{v:.3f}", va="center", ha="right", fontsize=7,
                color="white" if fam == "classical" else INK, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(op["model"])
    ax.set_xlabel(r"$F_1$ at the common operating point (clean-validation recall $=0.95$)")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="y", visible=False)
    handles = [mpl.patches.Patch(facecolor=BLUE, edgecolor=EDGE["classical"],
                                 label="Classical"),
               mpl.patches.Patch(facecolor=ORANGE, edgecolor=EDGE["deep"],
                                 label="Deep")]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    fig.tight_layout()
    save(fig, "fig05_detection_operating_point")


MODEL_ORDER = ["RandomForest", "XGBoost", "LightGBM", "GradientBoosting", "KNN",
               "MLP", "DecisionTree", "SVM", "CNN-1D", "LSTM", "BiLSTM", "CNN-LSTM",
               "Transformer", "TCN"]
FAMILY_OF = {m: ("classical" if i < 8 else "deep") for i, m in enumerate(MODEL_ORDER)}


def _gen_df():
    for p in (REPO / "generalization.csv", TAB / "generalization.csv"):
        if p.exists():
            g = pd.read_csv(p)
            if g["protocol"].nunique() >= 1 and "deep" in set(g["family"]):
                return g
    raise FileNotFoundError("no complete generalization.csv (needs deep rows)")


# ============================================================================
# F8 - Generalization: (a) cross-scenario heatmap  (b) leave-PRN spread
# ============================================================================
def fig_f8():
    g = _gen_df()
    cs = g[g.protocol == "cross_scenario"]
    lp = g[g.protocol == "leave_prn"]
    scen = ["ds2", "ds3", "ds7"]
    scen_lab = {"ds2": "ds2\n(overpowered)", "ds3": "ds3\n(matched-power)",
                "ds7": "ds7\n(phase-aligned)"}

    fig = plt.figure(figsize=(FULL_W, 15.5 * CM))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.42)

    # ---- (a) cross-scenario recall heatmap -------------------------------
    axa = fig.add_subplot(gs[0])
    M = cs.pivot_table(index="model", columns="holdout", values="recall").reindex(
        MODEL_ORDER)[scen]
    im = axa.imshow(M.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axa.set_xticks(range(len(scen)))
    axa.set_xticklabels([scen_lab[s] for s in scen])
    axa.set_yticks(range(len(MODEL_ORDER)))
    axa.set_yticklabels(MODEL_ORDER)
    for yi in range(len(MODEL_ORDER)):
        for xi in range(len(scen)):
            v = M.values[yi, xi]
            axa.text(xi, yi, f"{v:.2f}", ha="center", va="center", fontsize=6.8,
                     color="white" if v > 0.55 else INK)
    # family bracket: tick label color by family
    for lab, m in zip(axa.get_yticklabels(), MODEL_ORDER):
        lab.set_color(EDGE[FAMILY_OF[m]])
    axa.set_title("(a) Cross-scenario: detection recall on the held-out attack",
                  loc="left", fontsize=8.5)
    axa.grid(False)
    axa.tick_params(length=0)
    cb = fig.colorbar(im, ax=axa, fraction=0.025, pad=0.02)
    cb.set_label("Recall", fontsize=8); cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.5)

    # ---- (b) leave-PRN recall spread -------------------------------------
    axb = fig.add_subplot(gs[1])
    rng = np.random.default_rng(0)
    for xi, m in enumerate(MODEL_ORDER):
        vals = lp[lp.model == m]["recall"].values
        jit = rng.uniform(-0.16, 0.16, len(vals))
        axb.scatter(np.full(len(vals), xi) + jit, vals, s=13,
                    color=FAM_COLOR[FAMILY_OF[m]], edgecolor=EDGE[FAMILY_OF[m]],
                    linewidth=0.4, alpha=0.9, zorder=3)
        axb.plot([xi - 0.28, xi + 0.28], [np.median(vals)] * 2, color=INK,
                 linewidth=1.4, zorder=4)          # median tick
    axb.set_xticks(range(len(MODEL_ORDER)))
    axb.set_xticklabels(MODEL_ORDER, rotation=40, ha="right")
    for lab, m in zip(axb.get_xticklabels(), MODEL_ORDER):
        lab.set_color(EDGE[FAMILY_OF[m]])
    axb.set_ylabel("Recall on held-out satellite")
    axb.set_ylim(-0.03, 1.05)
    axb.set_title("(b) Leave-PRN: recall across 11 held-out satellites "
                  "(each point one satellite; bar = median)", loc="left", fontsize=8.5)
    axb.grid(axis="x", visible=False)
    handles = [mpl.patches.Patch(facecolor=BLUE, edgecolor=EDGE["classical"],
                                 label="Classical"),
               mpl.patches.Patch(facecolor=ORANGE, edgecolor=EDGE["deep"],
                                 label="Deep")]
    axb.legend(handles=handles, loc="lower left", frameon=False, ncol=2)
    save(fig, "fig08_generalization")


CORPUS = REPO / "data" / "processed" / "texbat_track_combined.csv"


# ============================================================================
# F2 - Spoofing-signature time-history: authentic (cleanStatic) vs spoofed (ds7)
# ============================================================================
def fig_f2():
    cols = ["scenario", "prn", "t_sec", "cn0_dbhz", "doppler_hz", "dll_discr"]
    df = pd.read_csv(CORPUS, usecols=cols)
    both = df[df.scenario.isin(["cleanstatic", "ds7"])]
    # pick the PRN with the most paired coverage and a strong signal
    cand = (both.groupby(["prn", "scenario"]).size().unstack(fill_value=0))
    cand = cand[(cand.get("cleanstatic", 0) > 100) & (cand.get("ds7", 0) > 100)]
    prn = int(both[both.prn.isin(cand.index)].groupby("prn")["cn0_dbhz"].median().idxmax())
    auth = both[(both.prn == prn) & (both.scenario == "cleanstatic")].sort_values("t_sec")
    spf = both[(both.prn == prn) & (both.scenario == "ds7")].sort_values("t_sec")

    specs = [("cn0_dbhz", r"$C/N_0$ (dB-Hz)"),
             ("doppler_hz", "Doppler (Hz)"),
             ("dll_discr", "DLL discriminator")]
    def gapped(d, c):
        # break the line where t_sec jumps (the discarded 110-150 s transition band),
        # so no straight interpolation is drawn across missing epochs
        t = d.t_sec.values.astype(float)
        y = d[c].values.astype(float)
        br = np.where(np.diff(t) > 1.0)[0] + 1
        return np.insert(t, br, np.nan), np.insert(y, br, np.nan)

    fig, axes = plt.subplots(3, 1, figsize=(FULL_W, 13 * CM), sharex=True)
    for ax, (c, lab) in zip(axes, specs):
        ax.axvspan(110, 150, color="#f0f0f0", zorder=0)          # discarded transition band
        ta, ya = gapped(auth, c)
        ts, ys = gapped(spf, c)
        ax.plot(ta, ya, color=BLUE, linewidth=1.1, label="Authentic (cleanStatic)")
        ax.plot(ts, ys, color=VERM, linewidth=1.1, label="Spoofed (ds7)")
        for xv in (110, 150):
            ax.axvline(xv, color=MUTED, linestyle=(0, (4, 3)), linewidth=0.8, zorder=1)
        ax.set_ylabel(lab)
        ax.set_xlim(0, 250)
        ax.text(130, 0.97, "takeover", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6.5, color=MUTED)
    axes[0].legend(loc="lower left", frameon=False, ncol=1)
    axes[-1].set_xlabel("Time (s)")
    fig.align_ylabels(axes)
    save(fig, f"fig02_signature_timehistory_prn{prn}")


# ============================================================================
# F3 - Feature distributions, genuine vs spoof (static corpus)
# ============================================================================
def fig_f3():
    feats = [("cn0_dbhz", r"$C/N_0$ (dB-Hz)"), ("doppler_hz", "Doppler (Hz)"),
             ("dll_discr", "DLL discriminator"), ("noise_cn0", "Noise floor (dB-Hz)"),
             ("i_prompt", r"$I_P$"), ("q_prompt", r"$Q_P$")]
    df = pd.read_csv(CORPUS, usecols=["label"] + [f for f, _ in feats])

    fig, axes = plt.subplots(2, 3, figsize=(FULL_W, 9.5 * CM))
    for ax, (c, lab) in zip(axes.ravel(), feats):
        g = df[df.label == 0][c].values
        s = df[df.label == 1][c].values
        lo, hi = np.nanpercentile(df[c].values, [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        ax.hist(g, bins=bins, density=True, color=BLUE, alpha=0.55, label="Genuine")
        ax.hist(s, bins=bins, density=True, color=VERM, alpha=0.55, label="Spoofed")
        ax.set_xlabel(lab); ax.set_yticks([])
        ax.set_xlim(lo, hi)
    axes[0, 0].legend(loc="upper right", frameon=False, fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Density")
    fig.tight_layout()
    save(fig, "fig03_feature_distributions")


# ============================================================================
# F9 - Adversarial realizability: the C/N0 to correlator-power manifold
# ============================================================================
def fig_f9():
    df = pd.read_csv(CORPUS, usecols=["cn0_dbhz", "i_prompt", "q_prompt", "label"])
    rng = np.random.default_rng(0)
    idx = rng.choice(len(df), min(6000, len(df)), replace=False)
    d = df.iloc[idx]
    logp = np.log10(np.clip(d.i_prompt.values ** 2 + d.q_prompt.values ** 2, 1e-9, None))
    cn0 = d.cn0_dbhz.values
    # fit the physical coupling C/N0 ~ a + b*log10(power) and its residual band
    b, a = np.polyfit(logp, cn0, 1)
    resid = cn0 - (a + b * logp)
    r_lo, r_hi = np.percentile(resid, [1, 99])
    xs = np.linspace(logp.min(), logp.max(), 100)

    # an attacker raising reported C/N0 without supplying the correlator power leaves
    # the band (unconstrained); the enforcer clips it back onto the band (enforced)
    sub = rng.choice(len(d), 500, replace=False)
    lp_a, cn0_a = logp[sub], cn0[sub]
    cn0_unc = cn0_a + 9.0                                   # raise C/N0 by 9 dB
    pred = a + b * lp_a
    cn0_enf = np.clip(cn0_unc, pred + r_lo, pred + r_hi)    # enforcer projection

    fig, ax = plt.subplots(figsize=(FULL_W, 8.5 * CM))
    ax.fill_between(xs, a + b * xs + r_lo, a + b * xs + r_hi, color="#e6e6e6",
                    zorder=0, label="Physically realizable band")
    ax.scatter(logp, cn0, s=4, color="#9a9a9a", alpha=0.5, zorder=1,
               label="Receiver observations", rasterized=True)
    ax.scatter(lp_a, cn0_unc, s=10, color=VERM, edgecolor=EDGE["deep"], linewidth=0.2,
               zorder=3, label="Unconstrained adversarial")
    ax.scatter(lp_a, cn0_enf, s=10, color=BLUE, edgecolor=EDGE["classical"],
               linewidth=0.2, zorder=2, label="Enforcer-constrained adversarial")
    ax.set_xlabel(r"$\log_{10}(I_P^2+Q_P^2)$ (correlator power)")
    ax.set_ylabel(r"Reported $C/N_0$ (dB-Hz)")
    ax.legend(loc="upper left", frameon=False, markerscale=1.6)
    ax.grid(True)
    fig.tight_layout()
    save(fig, "fig09_realizability")


MAT = HERE.parents[1] / "data" / "FGI_Data" / "out" / "trackData_cleanStatic_full.mat"


# ============================================================================
# F4 - Receiver validation: acquisition metric per PRN + one PRN's peak
# ============================================================================
def fig_f4():
    import scipy.io as sio
    m = sio.loadmat(str(MAT), struct_as_record=False, squeeze_me=True,
                    variable_names=["acqData", "settings"])
    acq = m["acqData"].gpsl1
    thr = float(getattr(m["settings"].gpsl1, "acqThreshold", 10))
    rows = [(int(c.SvId.satId), float(c.peakMetric), int(c.bFound)) for c in acq.channel]
    rows.sort(key=lambda r: -r[1])
    prn = [r[0] for r in rows]
    met = [r[1] for r in rows]
    acqd = [r[2] for r in rows]
    # strongest acquired PRN for panel (b)
    best = acq.channel[[int(c.SvId.satId) for c in acq.channel].index(prn[0])]
    spec = np.asarray(best.spec, dtype=float)
    chips = np.linspace(0, 1023, len(spec))

    # stacked layout: panel (a) gets the full text width so 32 PRN labels stay legible
    fig, (axa, axb) = plt.subplots(2, 1, figsize=(FULL_W, 11 * CM))
    x = np.arange(len(prn))
    cols = [BLUE if a else "#b0b0b0" for a in acqd]
    edg = [EDGE["classical"] if a else "#8a8a8a" for a in acqd]
    axa.bar(x, met, color=cols, edgecolor=edg, linewidth=0.6, width=0.8, zorder=3)
    axa.axhline(thr, color=VERM, linestyle=(0, (4, 3)), linewidth=1.0, zorder=4)
    axa.text(len(prn) - 0.5, thr + 1.8, "acquisition threshold", ha="right",
             va="bottom", fontsize=6.5, color=VERM)
    axa.set_xticks(x)
    axa.set_xticklabels(prn, fontsize=6, rotation=90)
    axa.set_xlim(-0.8, len(prn) - 0.2)
    axa.set_xlabel("Satellite (PRN)", labelpad=1)
    axa.set_ylabel("Acquisition metric")
    axa.set_title("(a) Acquisition metric per satellite", loc="left", fontsize=8.5)
    axa.grid(axis="x", visible=False)

    axb.plot(chips, spec / spec.max(), color=BLUE, linewidth=0.9)
    axb.set_xlabel("Code phase (chips)")
    axb.set_ylabel("Normalised correlation")
    axb.set_xlim(0, 1023)
    axb.set_title(f"(b) Acquisition of PRN {prn[0]}", loc="left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig04_receiver_validation")


# ============================================================================
# F10 - Slope chart: clean-performance rank vs adversarial-robustness rank
# ============================================================================
def fig_f10():
    op = pd.read_csv(TAB / "operating_point_recall95.csv")[["model", "family", "f1"]]
    bb = (pd.read_csv(TAB / "blackbox_boundary_all.csv")
          .groupby("model", as_index=False)["median_min_linf"].first())
    d = op.merge(bb, on="model")
    d["r_clean"] = d.f1.rank(ascending=False, method="first")          # 1 = best clean
    d["r_rob"] = d.median_min_linf.rank(ascending=False, method="first")  # 1 = most robust
    d["shift"] = d.r_rob - d.r_clean

    fig, ax = plt.subplots(figsize=(FULL_W, 11.5 * CM))
    for _, r in d.iterrows():
        big = abs(r["shift"]) >= 8                     # dramatic movers
        ax.plot([0, 1], [r.r_clean, r.r_rob],
                color=FAM_COLOR[r.family], linewidth=2.2 if big else 1.0,
                alpha=1.0 if big else 0.55, zorder=3 if big else 2,
                solid_capstyle="round")
        ax.scatter([0, 1], [r.r_clean, r.r_rob], s=22 if big else 12,
                   color=FAM_COLOR[r.family], edgecolor=EDGE[r.family],
                   linewidth=0.5, zorder=4)
        ax.text(-0.035, r.r_clean, f"{r.model}", ha="right", va="center",
                fontsize=7, color=INK, fontweight="bold" if big else "normal")
        ax.text(1.035, r.r_rob, f"{r.model}", ha="left", va="center",
                fontsize=7, color=INK, fontweight="bold" if big else "normal")
    ax.set_xlim(-0.42, 1.42); ax.set_ylim(len(d) + 0.6, 0.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Rank by clean $F_1$\n(1 = best detector)",
                        "Rank by robustness\n(1 = hardest to evade)"], fontsize=8)
    ax.set_ylabel("Rank")
    ax.set_yticks(range(1, len(d) + 1))
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", alpha=0.35)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    handles = [mpl.patches.Patch(facecolor=BLUE, edgecolor=EDGE["classical"], label="Classical"),
               mpl.patches.Patch(facecolor=ORANGE, edgecolor=EDGE["deep"], label="Deep")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.16),
              ncol=2, frameon=False)
    fig.tight_layout()
    save(fig, "fig10_rank_inversion")


# ============================================================================
# F11 - Parallel coordinates: four competing axes, no detector wins on all
# ============================================================================
def fig_f11():
    op = pd.read_csv(TAB / "operating_point_recall95.csv")
    bb = (pd.read_csv(TAB / "blackbox_boundary_all.csv")
          .groupby("model", as_index=False)["median_min_linf"].first())
    g = _gen_df()
    gen = (g[g.protocol == "cross_scenario"].groupby("model", as_index=False)["recall"]
           .mean().rename(columns={"recall": "gen"}))
    d = op.merge(bb, on="model").merge(gen, on="model")
    d["low_far"] = 1.0 - d.false_alarm_rate          # higher is better on every axis

    axes_spec = [("f1", "Clean $F_1$"), ("low_far", "1 - false-alarm rate"),
                 ("median_min_linf", "Adversarial robustness\n(min $L_\\infty$)"),
                 ("gen", "Cross-scenario recall")]
    cols = [c for c, _ in axes_spec]
    norm = d.copy()
    for c in cols:                                   # min-max per axis
        norm[c] = (d[c] - d[c].min()) / (d[c].max() - d[c].min())

    fig, ax = plt.subplots(figsize=(FULL_W, 9.5 * CM))
    xs = np.arange(len(cols))
    for _, r in norm.iterrows():
        ax.plot(xs, [r[c] for c in cols], color=FAM_COLOR[r.family], linewidth=1.3,
                alpha=0.75, marker="o", markersize=3.2, zorder=2)
    # label the two extremes of the robustness axis, which carry the story;
    # distinct dash patterns so the two highlighted detectors stay separable
    for name, ls, dy in [("GradientBoosting", "-", -0.035), ("SVM", (0, (5, 2)), 0.03)]:
        r = norm[norm.model == name].iloc[0]
        ax.plot(xs, [r[c] for c in cols], color=INK, linewidth=1.9, zorder=5,
                linestyle=ls, marker="o", markersize=4)
        ax.text(xs[-1] + 0.06, r[cols[-1]] + dy, name, fontsize=7.5,
                color=INK, ha="left", va="center", fontweight="bold")
    for x in xs:
        ax.axvline(x, color="#cccccc", linewidth=0.8, zorder=1)
    ax.set_xticks(xs); ax.set_xticklabels([lab for _, lab in axes_spec], fontsize=8)
    ax.set_xlim(-0.15, len(cols) - 0.42)
    ax.set_ylim(-0.06, 1.20)
    ax.set_ylabel("Normalised score (higher is better)")
    ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["worst", "", "best"])
    ax.grid(False)
    handles = [mpl.patches.Patch(facecolor=BLUE, edgecolor=EDGE["classical"], label="Classical"),
               mpl.patches.Patch(facecolor=ORANGE, edgecolor=EDGE["deep"], label="Deep"),
               mpl.lines.Line2D([0], [0], color=INK, lw=1.9, label="GradientBoosting"),
               mpl.lines.Line2D([0], [0], color=INK, lw=1.9, ls=(0, (5, 2)), label="SVM")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=4, frameon=False, fontsize=7.5, columnspacing=1.4)
    fig.tight_layout()
    save(fig, "fig11_tradeoff_parallel")


if __name__ == "__main__":
    print("Building figures ->", FIGD)
    fig_f4()
    fig_f10()
    fig_f11()
    fig_f5()
    fig_f6()
    fig_f8()
    fig_f2()
    fig_f3()
    fig_f9()
    print("done")
