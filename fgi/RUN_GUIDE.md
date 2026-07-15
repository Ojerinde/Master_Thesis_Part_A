# Paper 1 — TEXBAT Tracking and Feature Export (Run Guide)

Branch: `paper1-experiment`. Corpus is **static only**: cleanStatic (authentic) + ds2 + ds3 + ds7, with ds8 optional. ds5/ds6 (dynamic) are excluded.

Two environments are used:
- **MATLAB** (with FGI-GSRx) for tracking and feature export.
- **PowerShell** for the Python loader sanity check.

Paths below are literal — copy them as-is.

---

## Step 1 — Smoke test (MATLAB), ~1–2 min

Validates that FGI-GSRx reads the TEXBAT complex 16+16-bit format and acquires satellites, before committing to a multi-hour run. Run in the **MATLAB Command Window**:

```matlab
addpath(genpath('D:\BEIHANG UNIVERSITY\Research\FGI-GSRx'))
gsrx('D:\BEIHANG UNIVERSITY\Research\gnss_adversarial_research\fgi\user_texbat_ds7_smoke.txt')
```

**PASS looks like:** an acquisition plot appears, ~8–11 GPS PRNs report as acquired, C/N0 ≈ 45–50 dB-Hz, and tracking runs to 30 s with no error.

**If 0 satellites acquire:** the byte format is wrong (`complexData` / `sampleSize` / `iqSwap`). Stop and tell me — do not run the full tracks.

---

## Step 2 — Full-track the documented pair (MATLAB)

Only after the smoke passes. The configs use a **250 s window** (`msToProcess = 250e3`): FGI-GSRx tracking time scales roughly quadratically with window length, so 250 s runs in **~5–7 h/scenario** versus ~20 h for the full ~435 s, while still capturing genuine + post-onset spoof. Each writes a few-GB `trackData_*_full.mat` into `FGI_Data\out\`. Run them in a batch or overnight.

```matlab
gsrx('D:\BEIHANG UNIVERSITY\Research\gnss_adversarial_research\fgi\user_texbat_ds7_full.txt')
gsrx('D:\BEIHANG UNIVERSITY\Research\gnss_adversarial_research\fgi\user_texbat_cleanstatic_full.txt')
```

---

## Step 3 — Export features to CSV (MATLAB)

Reads the `trackData_*_full.mat` files and writes the labeled observable corpus. Missing scenarios are skipped, so this works with just ds7 + cleanStatic present.

```matlab
addpath('D:\BEIHANG UNIVERSITY\Research\gnss_adversarial_research\fgi')
export_texbat_track()
```

Output: `gnss_adversarial_research\data\processed\texbat_track_combined.csv`. It prints row counts and the genuine/spoof balance.

---

## Step 4 — Loader sanity check (PowerShell)

Confirms the Python side reads the new corpus. Run in **PowerShell**:

```powershell
cd 'D:\BEIHANG UNIVERSITY\Research\gnss_adversarial_research'
$env:PYTHONPATH = '.'
& C:\Python314\python.exe -c "from data.loader import load_texbat_track as L; df,f=L(); print('OK rows', len(df), 'features', len(f))"
```

Expect: it prints genuine/spoof counts, the scenario list, and 13 features.

**Stop here and send me the row counts and balance.** I will confirm the corpus is sound before you track the rest.

---

## Step 5 — Remaining scenarios (after the pair is validated)

```matlab
gsrx('D:\BEIHANG UNIVERSITY\Research\gnss_adversarial_research\fgi\user_texbat_ds2_full.txt')
gsrx('D:\BEIHANG UNIVERSITY\Research\gnss_adversarial_research\fgi\user_texbat_ds3_full.txt')
gsrx('D:\BEIHANG UNIVERSITY\Research\gnss_adversarial_research\fgi\user_texbat_ds8_full.txt')   % optional (near-duplicate of ds7)
```

Then re-run `export_texbat_track()` to fold them in.

---

## Notes

- Disk: raw `.bin` are ~40–45 GB each; each `trackData_*_full.mat` is a few GB. ~194 GB free is plenty for sequential runs.
- Do **not** re-run the Paper-2 `test_nav_feasibility` on a full recording until `run_fgi_nav.m`'s save is slimmed — that caused the 362 GB runaway.
- The smoke uses ds7's first 30 s (all authentic, pre-onset), so it tests acquisition/tracking only, not labeling.
