"""Plot a 2x6 grid of stars from the toy defocused-star set v0, from d = -1 to d = +1.

Usage:  python plot_toy_samples.py [path/to/toy_defocus_stars_v0.npz] [output.png]
Needs:  numpy, matplotlib
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # write to file, no window needed
import matplotlib.pyplot as plt

npz_path = sys.argv[1] if len(sys.argv) > 1 else "toy_defocus_stars_v0.npz"
out_path = sys.argv[2] if len(sys.argv) > 2 else "toy_samples_v0.png"

z = np.load(npz_path)
X = z["counts"][:, 0].astype(np.float32) / z["scale"]  # images, 0 to 1.40
d = z["d"]                                              # signed defocus, -1 to 1

# For each target d, take the image whose label is closest to it.
targets = [-1, -0.8, -0.6, -0.4, -0.2, -0.05, 0.05, 0.2, 0.4, 0.6, 0.8, 1]
idx = [int(np.abs(d - t).argmin()) for t in targets]

fig, ax = plt.subplots(2, 6, figsize=(12, 5.2))
for a, i in zip(ax.flat, idx):
    a.imshow(X[i], cmap="gray", vmin=0, vmax=1.4)  # same grey scale for every panel
    a.set_xticks([])
    a.set_yticks([])
    a.set_title(f"d = {d[i]:+.2f}   (#{i})", fontsize=10,
                color="#b03030" if d[i] < 0 else "#2050a0")  # red inside, blue outside focus

fig.suptitle("Toy defocused-star set v0: inside focus (d < 0, top row) "
             "and outside focus (d > 0, bottom row)", fontsize=11)
fig.text(0.5, 0.015, "Same radius at ±|d|; only the ring width differs (the invented sign rule). "
         "Grey scale 0–1.40, image = counts/200.", ha="center", fontsize=9, color="#555")
plt.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.07, wspace=0.08, hspace=0.28)
plt.savefig(out_path, dpi=120)
print(f"Saved {out_path} (images {idx})")
