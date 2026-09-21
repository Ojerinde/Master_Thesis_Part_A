"""
Kaggle runner for the two-track deep detector training.
=======================================================

Paste each block below into its own Kaggle notebook cell, or run this file
whole as a single cell. Settings: Accelerator = GPU T4 x2 (or None for CPU),
Internet = ON (needed to clone the repo).

Reported results must come from ONE device for every model. Running the same
configuration on both CPU and GPU is a reproducibility check, not a comparison
to publish: any difference between them is floating-point accumulation order,
not a property of the detectors.
"""

# ---------------------------------------------------------------- cell 1 ----
# Clone the code. The CSV is gitignored in the repo, so it arrives separately
# as a Kaggle Dataset and is copied into the path config/paths.py expects.
CELL_1 = r"""
!rm -rf /kaggle/working/Master_Thesis_Part_A
!git clone --depth 1 https://github.com/Ojerinde/Master_Thesis_Part_A.git /kaggle/working/Master_Thesis_Part_A
%cd /kaggle/working/Master_Thesis_Part_A
!git log -1 --format="commit %h  %ad  %s" --date=short
"""

# ---------------------------------------------------------------- cell 2 ----
# Put the corpus where the loader looks for it, and prove it is the same file
# as the local one by printing its md5 and row count.
CELL_2 = r"""
import os, glob, hashlib, shutil

src = glob.glob('/kaggle/input/**/texbat_track_combined.csv', recursive=True)
assert src, 'Upload texbat_track_combined.csv as a Kaggle Dataset and attach it'
src = src[0]
dst = '/kaggle/working/Master_Thesis_Part_A/data/processed/texbat_track_combined.csv'
os.makedirs(os.path.dirname(dst), exist_ok=True)
shutil.copy(src, dst)

h = hashlib.md5(open(dst,'rb').read()).hexdigest()
print('corpus :', src)
print('md5    :', h)
print('size   : %.1f MB' % (os.path.getsize(dst)/1e6))
"""

# ---------------------------------------------------------------- cell 3 ----
# Environment, recorded so the methods section can state it exactly.
CELL_3 = r"""
import torch, numpy, sklearn, platform, sys
print('python      ', sys.version.split()[0])
print('torch       ', torch.__version__)
print('numpy       ', numpy.__version__)
print('sklearn     ', sklearn.__version__)
print('cuda avail  ', torch.cuda.is_available())
print('device      ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
print('platform    ', platform.platform())
"""

# ---------------------------------------------------------------- cell 4 ----
# The main run: both tracks, six architectures, three seeds each.
CELL_4 = r"""
%cd /kaggle/working/Master_Thesis_Part_A
!python experiments/02x_two_track.py --window 20 --outdir /kaggle/working/out
"""

# ---------------------------------------------------------------- cell 5 ----
# Sensitivity runs a reviewer will ask for: window length, and the control that
# gives the windowed models a comparable number of gradient steps.
CELL_5 = r"""
!python experiments/02x_two_track.py --window 10 --tracks windowed --outdir /kaggle/working/out_w10
!python experiments/02x_two_track.py --window 40 --tracks windowed --outdir /kaggle/working/out_w40
!python experiments/02x_two_track.py --window 20 --tracks windowed --train-stride 5 --outdir /kaggle/working/out_stride5
"""

# ---------------------------------------------------------------- cell 6 ----
# Collect everything into one zip under /kaggle/working so the notebook output
# is a single download.
CELL_6 = r"""
import shutil, os, glob
os.makedirs('/kaggle/working/results_bundle', exist_ok=True)
for d in ['out', 'out_w10', 'out_w40', 'out_stride5']:
    p = f'/kaggle/working/{d}'
    if os.path.isdir(p):
        shutil.copytree(p, f'/kaggle/working/results_bundle/{d}', dirs_exist_ok=True)
shutil.make_archive('/kaggle/working/two_track_results', 'zip',
                    '/kaggle/working/results_bundle')
print('bundle: /kaggle/working/two_track_results.zip')
for f in sorted(glob.glob('/kaggle/working/results_bundle/**/*.csv', recursive=True)):
    print('  ', f.replace('/kaggle/working/results_bundle/', ''))
"""

# ---------------------------------------------------------------- cell 7 ----
# Print the summary inline so it is visible in the committed notebook even
# before the zip is downloaded.
CELL_7 = r"""
import pandas as pd, glob
for f in sorted(glob.glob('/kaggle/working/out*/summary_both_*.csv')
                + glob.glob('/kaggle/working/out*/summary_*_w*.csv')):
    print('='*70); print(f)
    print(pd.read_csv(f).to_string(index=False))
"""

if __name__ == "__main__":
    for i, c in enumerate([CELL_1, CELL_2, CELL_3, CELL_4, CELL_5, CELL_6, CELL_7], 1):
        print(f"\n{'#'*70}\n# CELL {i}\n{'#'*70}{c}")
