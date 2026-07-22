"""Consistency sweep: recompute every number the manuscript cites from the source
CSVs and print it as the single source of truth, so the text/Table 1/figures can be
checked against it. Run: python verify_numbers.py"""
from pathlib import Path
import pandas as pd
import numpy as np

REPO = Path(__file__).resolve().parents[1].parent / "gnss_adversarial_research"
TAB = REPO / "results" / "tables"


def sec(t): print("\n" + "=" * 66 + f"\n{t}\n" + "=" * 66)


# --- clean detection (Table 1 + inline) --------------------------------------
sec("CLEAN DETECTION  (operating_point_recall95.csv)")
op = pd.read_csv(TAB / "operating_point_recall95.csv")
print(op[["model", "family", "tau", "recall", "false_alarm_rate", "precision",
          "f1", "auc_roc"]].round(3).to_string(index=False))
print(f"\nF1  range: {op.f1.min():.3f} ({op.loc[op.f1.idxmin(),'model']}) .. "
      f"{op.f1.max():.3f} ({op.loc[op.f1.idxmax(),'model']})")
print(f"AUC range: {op.auc_roc.min():.3f} ({op.loc[op.auc_roc.idxmin(),'model']}) .. "
      f"{op.auc_roc.max():.3f} ({op.loc[op.auc_roc.idxmax(),'model']})")
print(f"FAR range: {op.false_alarm_rate.min():.3f} ({op.loc[op.false_alarm_rate.idxmin(),'model']}) .. "
      f"{op.false_alarm_rate.max():.3f} ({op.loc[op.false_alarm_rate.idxmax(),'model']})")

sec("SECOND OPERATING POINT  (operating_point_far05.csv): FAR<=0.05 feasibility")
far = pd.read_csv(TAB / "operating_point_far05.csv")
print("far_ceiling_met == False (cannot reach FAR<=0.05 on val):",
      far.loc[~far.far_ceiling_met, "model"].tolist())
print(far[["model", "far_ceiling_met", "recall", "false_alarm_rate"]].round(3).to_string(index=False))

# --- adversarial: decision-based min-Linf ------------------------------------
sec("ADVERSARIAL  (blackbox_boundary_all.csv): median min-Linf, sorted (fragility)")
bb = pd.read_csv(TAB / "blackbox_boundary_all.csv")
d = bb.groupby(["model", "family"], as_index=False)["median_min_linf"].first().sort_values("median_min_linf")
print(d.to_string(index=False))
cl = d[d.family == "classical"].median_min_linf
dp = d[d.family == "deep"].median_min_linf
print(f"\nmost fragile: {d.iloc[0]['model']} {d.median_min_linf.min():.4f}")
print(f"classical excl KNN excl GB range: "
      f"{d[(d.family=='classical')&(~d.model.isin(['KNN','GradientBoosting']))].median_min_linf.min():.4f} .. "
      f"{d[(d.family=='classical')&(~d.model.isin(['KNN','GradientBoosting']))].median_min_linf.max():.4f}")
print(f"deep range: {dp.min():.4f} .. {dp.max():.4f}")
print(f"KNN: {d[d.model=='KNN'].median_min_linf.iloc[0]:.4f}")

sec("ADVERSARIAL  (adversarial_full_oppoint.csv): domain-attack max ASR")
ad = pd.read_csv(TAB / "adversarial_full_oppoint.csv")
dom = ad[ad.attack.isin(["DLSA", "SNA", "TPA"])]
print(f"DLSA/SNA/TPA ASR: min {dom.asr.min():.4f}  max {dom.asr.max():.4f}  "
      f"(max at {dom.loc[dom.asr.idxmax(),'model']}/{dom.loc[dom.asr.idxmax(),'attack']}"
      f"@eps{dom.loc[dom.asr.idxmax(),'eps']})")

# --- generalization ----------------------------------------------------------
sec("GENERALIZATION  (generalization.csv)")
gpath = REPO / "generalization.csv"
if not gpath.exists():
    gpath = TAB / "generalization.csv"
g = pd.read_csv(gpath)
cs = g[g.protocol == "cross_scenario"]
lp = g[g.protocol == "leave_prn"]
print("cross-scenario mean recall/F1 by family:")
print(cs.groupby("family")[["recall", "f1"]].mean().round(4).to_string())
print("\nds2 (worst scenario) recall:")
ds2 = cs[cs.holdout == "ds2"].sort_values("recall")
print(f"  min: {ds2.iloc[0]['model']} {ds2.iloc[0]['recall']:.3f} ;  "
      f"best classical {ds2[ds2.family=='classical'].recall.max():.3f} ;  "
      f"best deep {ds2[ds2.family=='deep'].recall.max():.3f}")
ds7 = cs[cs.holdout == "ds7"]
print(f"ds7 best deep: {ds7[ds7.family=='deep'].sort_values('recall').iloc[-1]['model']} "
      f"{ds7[ds7.family=='deep'].recall.max():.3f} ; best classical "
      f"{ds7[ds7.family=='classical'].recall.max():.3f}")
print("\nleave-PRN worst (min recall) per model:")
w = lp.groupby("model")["recall"].min().sort_values()
print(w.round(4).head(6).to_string())

# --- latency -----------------------------------------------------------------
sec("LATENCY  (latency_all13.csv)")
lat = pd.read_csv(TAB / "latency_all13.csv")
print(f"inference us: {lat.infer_us.min():.3f} ({lat.loc[lat.infer_us.idxmin(),'model']}) .. "
      f"{lat.infer_us.max():.3f} ({lat.loc[lat.infer_us.idxmax(),'model']})")
print(f"attack-gen/inference ratio max: {lat.dlsa_ratio.max():.4f}")
