"""F7 - SQM multi-correlator correlation function, authentic vs takeover (ds7, PRN 23).
Reads the real multi-correlator trackData produced by FGI-GSRx
(enableMultiCorrelatorTracking, 17 taps over +/-2 chips). Journal style, matches
make_figures.py. Run: python make_f7.py  ->  figures/fig07_sqm.pdf + .png
"""
from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGD = HERE / "figures"; FIGD.mkdir(exist_ok=True)
MAT = HERE.parents[1] / "data" / "FGI_Data" / "out" / "trackData_ds7_full.mat"

BLUE, VERM, GREEN, INK, MUTED, GRID = "#0072B2", "#D55E00", "#009E73", "#1a1a1a", "#5c5c5c", "#dddddd"
CM = 1 / 2.54; FULL_W = 16 * CM
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": MUTED,
})

PRN, T_AUTH, T_SPF = 23, (30, 105), (155, 220)   # authentic (t<110) vs spoofed (t>150)

m = sio.loadmat(MAT, struct_as_record=False, squeeze_me=True)
td = m["trackData"].gpsl1
ch = {getattr(c.SvId, "satId", None): c for c in td.channel}[PRN]
off = np.asarray(ch.mulCorrFingers, float)
mag = np.abs(np.asarray(ch.mulCorrFingersOut, float))
fps = mag.shape[0] / 250.0


def window(t0, t1):
    v = mag[int(t0 * fps):int(t1 * fps)].mean(axis=0)
    return v / v.max()


auth, spf = window(*T_AUTH), window(*T_SPF)
diff = spf - auth

fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, FULL_W * 0.44))

a.plot(off, auth, color=BLUE, marker="o", ms=3.2, label="Authentic (t < 110 s)")
a.plot(off, spf, color=VERM, ls="--", marker="s", ms=3.0,
       label="Spoofed (t > 150 s)")
a.set_xlabel("Code offset (chips)")
a.set_ylabel("Normalised correlation magnitude")
a.set_xticks(np.arange(-2, 2.01, 1)); a.set_xlim(-2.1, 2.1); a.set_ylim(0, 1.06)
a.legend(loc="upper right", frameon=False, handlelength=1.6)
a.set_title("(a) Correlation function", loc="left")

b.axhline(0, color=MUTED, lw=0.6)
b.plot(off, diff, color=GREEN, marker="D", ms=3.0)
b.fill_between(off, 0, diff, color=GREEN, alpha=0.15)
b.set_xlabel("Code offset (chips)")
b.set_ylabel(r"Correlation difference (spoofed $-$ authentic)")
b.set_xticks(np.arange(-2, 2.01, 1)); b.set_xlim(-2.1, 2.1)
b.set_ylim(-0.03, 0.03)
b.set_title("(b) Distortion residual", loc="left")

fig.tight_layout(w_pad=2.0)
for ext in ("pdf", "png"):
    fig.savefig(FIGD / f"fig07_sqm.{ext}")
plt.close(fig)
print(f"PRN {PRN}: peak both at offset {off[auth.argmax()]:+.2f} chip; "
      f"off-peak floor rise {diff[np.abs(off) >= 1.25].mean():+.4f}")
print("wrote figures/fig07_sqm.pdf + .png")
