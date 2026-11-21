"""Plot 12 stars from dataset v1, from the most negative to the most positive dz.
Usage: python plot_samples_v1.py [defocus_stars_v1.npz] [out.png]   Needs numpy, matplotlib."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "defocus_stars_v1.npz"
out = sys.argv[2] if len(sys.argv) > 2 else "samples_v1.png"
z = np.load(path)
X, dz, snr, fw = z["adu"], z["dz_um"], z["ring_snr"], z["seeing_fwhm_px"]
targets = [-1200, -1000, -800, -600, -300, -80, 80, 300, 600, 800, 1000, 1200]
idx = [int(np.abs(dz - t).argmin()) for t in targets]

fig, ax = plt.subplots(2, 6, figsize=(13, 5.6))
for a, i in zip(ax.flat, idx):
    a.imshow(X[i], cmap="gray", origin="lower")       # each panel auto-scaled, like the real frame
    a.set_xticks([]); a.set_yticks([])
    a.set_title(f"dz = {dz[i]:+.0f} µm\nSNR {snr[i]:.1f}, seeing {fw[i]:.1f} px", fontsize=9,
                color="#b03030" if dz[i] < 0 else "#2050a0")
fig.suptitle("Dataset v1: inside focus (dz < 0, top row) and outside focus (dz > 0, bottom row), raw ADU",
             fontsize=11)
plt.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.02, wspace=0.08, hspace=0.32)
plt.savefig(out, dpi=110)
print(f"Saved {out} (images {idx})")
