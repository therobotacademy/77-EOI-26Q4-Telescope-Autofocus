# Student Self-Study Guide — Week 1 to Week 2 (Single-Image Telescope Autofocus)

Welcome to your Week 1 → Week 2 independent work guide. You are looking at the same problem through **two lenses at once**:

- **The optical physics lens:** why a defocused star encodes the **direction** of focus error ($\operatorname{sign}(\Delta z)$) *only* when something breaks symmetry inside vs. outside focus — and how Fourier optics turns pupil geometry and wavefront phase into raw camera ADU counts.
- **The deep learning engineering lens:** how to inspect code line by line, avoid synthetic shortcuts and `test_idx` leakage, build a clean classical **Half-Flux Radius (HFR)** baseline, and work reliably inside CPU-only training budgets.

By next week's class, your goal is to close **Gate 1** (prove the sign is physically visible for *your* telescope) and **Gate 2** (generate your calibrated dataset and measure the classical HFR baseline error per $|\Delta z|$ bin).

---

## What You Will Accomplish in the Next 7 Days

- [ ] **Day 1:** Verify your environment (Git LFS binary archives, Python stack) and reproduce the baseline CNN sanity check on the placeholder physical dataset.
- [ ] **Day 2:** Understand the optical physics by hand (micro-sandbox calculations) and lock the **4 open Week-1 decisions** for your real telescope and camera.
- [ ] **Day 3:** Dissect [`star_sim.py`](../Fourier-optics%20simulator/star_sim.py), plug in your real optics and your exact `"preprocesada"` function, and pass **Gate 1**.
- [ ] **Day 4:** Rebuild your calibrated 20,000-star dataset safely (clearing stale shards first) and visually audit the generated frames.
- [ ] **Day 5:** Implement and run the classical **HFR baseline** per $|\Delta z|$ bin to pass **Gate 2**.
- [ ] **Day 6:** Prepare real-sky validation — capture a raw FITS focuser sweep (`FOCPOS`) or test your cut-out + preprocessing + HFR pipeline on [MILAN Sky Survey](https://zenodo.org/records/7304607) FITS frames.
- [ ] **Day 7:** Package your deliverables and complete the self-check quiz before next week's session.

## Technical Prerequisites & Repository Map

- **Software stack:** Python 3, `numpy`, `torch` (CPU is enough), `matplotlib`, plus `astropy` (for reading real FITS headers and pixel arrays) and Git LFS (`git lfs`).
- **Hardware budget reference** (from the course sandbox benchmarks in [Problem Statement & Plan](../Single-Image%20Telescope%20Autofocus%20—%20Problem%20Statement%20&%20Plan.md)): on 2 Xeon CPU cores without a GPU, `64×64` cut-outs train at `~340 images/s` (`~1 min` per 20k-image epoch), whereas `128×128` drops to `~60 images/s` (`~5.5 min/epoch`). Keep cut-outs at `64×64` (`≤ 96×96`) and keep experiments under ~10 minutes.
- **Key repository documents (relative to this folder):**
  - Project overview: [`README.md`](../README.md)
  - Problem spec, metrics, 6-week gates & open decisions: [`Single-Image Telescope Autofocus — Problem Statement & Plan.md`](<../Single-Image%20Telescope%20Autofocus%20—%20Problem%20Statement%20&%20Plan.md>)
  - How all materials were created (12-prompt log): [`WORKFLOW_PROCEDURE.md`](../WORKFLOW_PROCEDURE.md)
  - Non-physical toy benchmark (smoke tests only): [`Benchmark v0/README.md`](<../Benchmark%20v0/README.md>) and [`Dataset Card`](<../Dataset%20Card%20—%20Toy%20Defocused-Star%20Set%20%28benchmark%20v0%29.md>)
  - Physical simulator & Dataset v1: [`Fourier-optics simulator/README_v1.md`](<../Fourier-optics%20simulator/README_v1.md>)

---

## Day 1 — Environment Setup & Reproducing the Baseline

Before editing any physics or training code, verify that Git LFS downloaded the actual `.npz` archives rather than tiny 3-line pointer text files, and reproduce the existing sanity check.

> [!CAUTION]
> **Check Git LFS before running Python.** Both `.npz` datasets are stored in Git LFS (`~57 MiB` and `~104 MiB`). If you cloned without Git LFS installed or initialized, `defocus_stars_v1.npz` will only be ~130 bytes long and `np.load(...)` will crash with `BadZipFile`.

Run from the repository root:

```bash
git lfs pull
python -c "import os; print('v0 bytes:', os.path.getsize('Benchmark v0/toy_defocus_stars_v0.npz')); print('v1 bytes:', os.path.getsize('Fourier-optics simulator/defocus_stars_v1.npz'))"
```

Expected sizes: `59999877` bytes (~57 MiB) for `toy_defocus_stars_v0.npz` and `108560366` bytes (~104 MiB) for `defocus_stars_v1.npz`.

Next, run the existing visual inspection and 4-epoch CNN baseline from inside [`Fourier-optics simulator/`](<../Fourier-optics%20simulator/README_v1.md>):

```bash
cd "Fourier-optics simulator"
python plot_samples_v1.py defocus_stars_v1.npz samples_v1.png
python train_check_v1.py defocus_stars_v1.npz 4
```

**Expected output from `train_check_v1.py` (4 epochs, ~183 s on 2 CPU cores):**
- Overall test MAE around `148 µm` and overall sign accuracy around `88.9%`.
- Binned sign accuracy should show the characteristic near-focus dip and strong far-focus recovery:
  - `0 – 200 µm`: `~58%`
  - `200 – 400 µm`: `~90%`
  - `400 – 800 µm`: `~95%`
  - `> 800 µm`: `~98%`

> [!CAUTION]
> **Stay out of `Benchmark v0/` for autofocus experiments.** [`Benchmark v0/toy_defocus_stars_v0.npz`](<../Benchmark%20v0/README.md>) achieves `98.2%` sign accuracy in 3 epochs only because of an invented rule (`w = 1.5 + 0.5·d` inside [`make_toy_defocus_stars.py`](<../Benchmark%20v0/make_toy_defocus_stars.py>)) that makes inside-focus rings thinner than outside-focus rings. Use it solely if you want to verify PyTorch CPU throughput in 7.5 seconds.

---

## Day 2 — Optical Physics Intuition, Micro-Sandbox & Your 4 Week-1 Decisions

### Why Is the Sign Hard?

Think of a classical V-curve focuser sweep as measuring the width of a spotlight beam on a wall as you move a magnifying lens forward and backward. Far on either side of focus, the spotlight forms a wide circle of the same diameter. Looking at **one frozen snapshot** of a clean circle tells you *how far* you are from focus, **not** whether to turn the focuser knob clockwise or counter-clockwise.

In real reflecting or imperfectly corrected telescopes, **asymmetric wavefront aberrations** break that mirror symmetry:
- With positive **spherical aberration** (`spherical_waves = 0.4` in `star_sim.py`, Zernike $Z_{11} = 6\rho^4 - 6\rho^2 + 1$), marginal rays (near the outer edge of the mirror) focus at a slightly different plane than paraxial rays (near the inner edge).
- As a result, at equal $|\Delta z|$, **inside focus (`dz < 0`)** produces a crisper ring with a darker central shadow, while **outside focus (`dz > 0`)** shifts more energy inward, producing a softer, partially filled donut.

Look closely at the two reference plots already generated in [`Fourier-optics simulator/`](<../Fourier-optics%20simulator/README_v1.md>):

![Simulator v1 calibration vs real frame](../Fourier-optics%20simulator/sim_vs_real_v1.png)

![Dataset v1 raw ADU samples across defocus](../Fourier-optics%20simulator/samples_v1.png)

**Checkpoint question:** Compare the top row (`dz < 0`) against the bottom row (`dz > 0`) in `samples_v1.png` at `±1000 µm` vs. `±80 µm`. Why can *your* eye tell top from bottom at `±1000 µm` while `±80 µm` looks like two nearly identical blobs — and how does that explain why `train_check_v1.py` drops from `98%` sign accuracy above `800 µm` to `58%` below `200 µm`?

---

### Pencil-and-Paper Micro-Sandbox: Checking Whether a Donut Fits in `64×64`

Before trusting any simulator parameters for *your* telescope, check the focal-plane geometry by hand. Let's work through the exact numbers currently set in [`star_sim.py`](<../Fourier-optics%20simulator/star_sim.py>) (lines 25–71, placeholder `200 mm f/5` telescope, `λ = 0.55 µm`, `pixel_um = 3.76`, `pupil_px = 256`, `grid = 512`, `defocus_waves = ±11`):

1. **Focal ratio ($N$):**
   $$N = \frac{f}{D} = \frac{1000\text{ mm}}{200\text{ mm}} = 5.0$$

2. **Focal-plane shift per 1 wave peak-to-valley of defocus (`Telescope.dz_per_wave()`):**
   $$\Delta z_{\text{per wave}} = 8 N^2 \lambda = 8 \times (5.0)^2 \times 0.55\ \mu\text{m} = 200 \times 0.55 = 110.0\ \mu\text{m/wave}$$
   Over `defocus_waves ∈ [-11, +11]`, the full focal error range is:
   $$\Delta z \in [-11 \times 110,\ +11 \times 110] = [-1{,}210,\ +1{,}210]\ \mu\text{m}$$

3. **Camera sampling vs. FFT sampling (`zoom` and crop `side`):**
   - Camera pixel size in diffraction ($\lambda/D$) units:
     $$\texttt{cam\_px\_lod} = \frac{\text{pixel\_um}}{\lambda \cdot N} = \frac{3.76}{0.55 \times 5.0} = \frac{3.76}{2.75} \approx 1.3673\ \lambda/D\text{ per px}$$
   - FFT pixel size in $\lambda/D$ units:
     $$\texttt{fft\_px\_lod} = \frac{\texttt{pupil\_px}}{\texttt{grid}} = \frac{256}{512} = 0.50\ \lambda/D\text{ per px}$$
   - FFT pixels per camera pixel:
     $$\texttt{zoom} = \frac{1.36727}{0.50} \approx 2.7345$$
   - Cropped FFT box for a `64×64` cut-out:
     $$\texttt{side} = \lceil 64 \times 2.73455 \rceil = \lceil 175.01 \rceil = 176\ \text{px} \le 512\ (\texttt{grid})$$
     *(If your camera has larger pixels or a faster focal ratio so `side > grid`, cropping fails — you would need to raise `grid` in `StarSimulator`!)*

4. **Geometric outer donut radius at $|\Delta z| = 1{,}210\ \mu\text{m}$:**
   At focal ratio $N$, an $f/N$ light cone opens with half-angle $\approx 1/(2N)$, so a defocus $|\Delta z|$ creates a geometric blur disk of outer radius:
   $$R_{\text{outer}} \approx \frac{|\Delta z|}{2 N \cdot \text{pixel\_um}} = \frac{1{,}210}{2 \times 5.0 \times 3.76} = \frac{1{,}210}{37.6} \approx 32.18\ \text{px}$$
   Because a `64×64` cut-out only extends `32 px` from the center (and stars also have `±3 px` center jitter plus `6–12 px` seeing blur), `±11` waves is right at the physical edge of the `64×64` frame — which is why Phase 6 (`P-11`) narrowed `defocus_waves` from its wider initial test to `±11`!

> [!TIP]
> **Re-run those four lines of arithmetic for your own telescope.** Plug your own `aperture_mm`, `focal_mm`, and `pixel_um` into the formulas above *before* generating 20,000 images so you know both `defocus_waves` (to keep the largest donut inside `64×64`) and `side <= grid`.

---

### Lock Down the 4 Open Decisions from Week 1

In the [Problem Statement & Plan](<../Single-Image%20Telescope%20Autofocus%20—%20Problem%20Statement%20&%20Plan.md>), four decisions were left open for Week 1. Fill in your actual telescope numbers and commit to them before Wednesday/Thursday:

| # | Decision | Placeholder in `v1` | Your Value for This Course |
|---|---|---|---|
| 1 | **Telescope & camera** (`aperture_mm`, `focal_mm`, `obstruction`, `spider_vanes`, `pixel_um`, `gain`, `bias`) | `200 mm`, `1000 mm` (`f/5`), `0.30`, `0` vanes, `3.76 µm`, `1.0`, `100` | `[YOUR TELESCOPE & CAMERA]` |
| 2 | **Canonical unit for $\Delta z$** | `µm` at the focal plane (`dz_um`, `110 µm/wave`) | `[µm / focuser steps / waves]` *(if steps, record `µm_per_step`)* |
| 3 | **Target success thresholds** | Proposal: `≥ 95%` sign accuracy outside near-focus zone; `1–2` autofocus moves; beat HFR MAE | `[CONFIRM OR ADJUST]` |
| 4 | **Real focuser sweep availability** | Pending (only 1 screenshot used so far) | `[YES (target date) / NO — use MILAN + own single FITS]` |

If you already have **raw FITS frames** (including the real frame behind your screenshot and your `"preprocesada"` script), inspect their true 16-bit headers and statistics first — do not rely on screenshot grayscale levels:

```python
# inspect_fits.py — run on one or more raw 16-bit FITS frames from your camera
import numpy as np
from astropy.io import fits

path = "path/to/your_raw_frame.fits"
with fits.open(path) as hdul:
    img = hdul[0].data.astype(np.float32)
    hdr = hdul[0].header

for key in ("FOCPOS", "FILTER", "EXPTIME", "GAIN", "EGAIN", "XPIXSZ", "FOCALLEN", "APTDIA"):
    if key in hdr:
        print(f"{key}: {hdr[key]}")

print(f"Shape: {img.shape}, min/med/max ADU: {img.min():.1f} / {np.median(img):.1f} / {img.max():.1f}")
# If you crop a 64x64 star cutout `cut`:
# border = np.concatenate([cut[:4].ravel(), cut[-4:].ravel(), cut[:, :4].ravel(), cut[:, -4:].ravel()])
# sky_med, bg_std = np.median(border), border.std()
```

---

## Day 3 — Under the Hood: Dissecting `star_sim.py` & `train_check_v1.py`, Plus Passing Gate 1

### Line-by-Line Walkthrough of [`star_sim.py`](<../Fourier-optics%20simulator/star_sim.py>)

Open `Fourier-optics simulator/star_sim.py` alongside these notes:

- **Lines 24–60 (`Telescope` & `Ranges` dataclasses):**
  `Telescope` holds deterministic hardware geometry (`aperture_mm`, `focal_mm`, `obstruction`, `spider_vanes`, `vane_width`, `wavelength_um`, `pixel_um`, `gain_e_per_adu`, `bias_adu`, `saturation_adu`, `cutout`, `spherical_waves`). `Ranges` holds per-star randomized conditions drawn uniformly for every image (`defocus_waves`, `seeing_fwhm_px`, `moffat_beta`, `d_over_r0`, `low_order_waves`, `center_jitter_px`, `sky_e`, `read_noise_e`, `ring_snr`).
- **Lines 62–103 (`StarSimulator.__init__`):**
  Constructs the normalized pupil coordinate grid `rho` ($\rho \le 1$ is the aperture edge), builds the binary amplitude mask `amp` (`obstruction <= r <= 1`, masking out `spider_vanes` half-lines if `spider_vanes > 0`), precomputes the six aberration shapes in `self.Z` (`defocus`, `spherical`, `astig0`, `astig45`, `comax`, `comay`), and builds the spatial-frequency filter `fr ** (-11/6)` (`self.kolmo`) for atmospheric Kolmogorov turbulence.
- **Lines 107–125 (`_phase_screens` & `_moffat_kernel`):**
  `_phase_screens` filters white Gaussian noise with `self.kolmo`, subtracts piston over the illuminated pupil, and rescales phase RMS to match Noll's `(D/r0)^(5/6)` scaling in waves. `_moffat_kernel` builds normalized `47×47` Moffat seeing kernels parameterized by `seeing_fwhm_px` and `moffat_beta`.
- **Lines 126–178 (`StarSimulator.batch`):**
  1. Draws or overrides (`fixed`) `defocus_waves` and converts to `dz = W_def * t.dz_per_wave()`.
  2. Adds `t.spherical_waves * Z["spherical"]`, random astigmatism/coma, and the Kolmogorov turbulence screen to form wavefront `W`.
  3. Applies sub-pixel center jitter (`sh`) directly in Fourier space as a linear pupil phase `tilt`, then forms the complex pupil field `amp * exp(2j * pi * (W + tilt))`.
  4. Computes the diffraction PSF `|FFT(field)|²`, crops `side × side`, bins down to `64×64` camera pixels with `F.adaptive_avg_pool2d`, and convolves with the Moffat seeing kernel. If `clean=True`, it stops here and returns the noiseless unit-flux image (ideal for Gate 1 checks!).
  5. Otherwise, scales total star `flux` so that mean signal on pixels above 25% of peak equals `ring_snr * sqrt(sky + read_noise²)`, simulates Poisson shot noise + Gaussian read noise, divides by `gain`, adds `bias_adu`, rounds, and clips to `[0, saturation_adu]`.
- **Lines 190–199 (`preprocess` placeholder):**
  Currently a **guess** (`(adu - bias - med) / mad`, scaled using the `99.5`th percentile).

> [!TIP]
> **Replace `preprocess()` with your real `"preprocesada"` function immediately.** A CNN trained on one contrast stretch and evaluated on another will fail on real stars even if the simulator physics are perfect. Put your exact preprocessing logic in `preprocess()` and call that *exact same function* during synthetic training, validation, and real-sky inference.

### Line-by-Line Walkthrough of [`train_check_v1.py`](<../Fourier-optics%20simulator/train_check_v1.py>)

- **Lines 10–20:** Loads `adu` as `float32`, estimates per-image background median and std from a 4-pixel border around the cut-out (`X = ((adu - med) / sd)[:, None] / 10`), normalizes the target `Y = dz / max(|dz|)`, and reads `train_idx` (`16,000`) and `test_idx` (`2,000`).
- **Lines 22–28:** Defines a compact 4-block CNN (`1 → 32 → 64 → 128 → 128`, each `3×3 Conv + ReLU + 2×2 MaxPool`, except block 4 uses `AdaptiveAvgPool2d(1)`), followed by `128 → 64 → 1` linear head and `Adam(1e-3)`.
- **Lines 30–48:** Trains with `L1Loss` for `epochs` (`4` by default) and prints overall MAE, sign accuracy, and per-`|dz|` bin breakdown.

**Notice two things missing in `train_check_v1.py` that you should fix in your own scripts from Day 4 onward:**
1. It evaluates on `test_idx` instead of `val_idx` (`2,000` validation stars are already saved in the `.npz`!). Keep `test_idx` untouched during model tuning, preprocessing changes, and architecture comparisons — check `val_idx` while iterating.
2. It uses raw 4-px border normalization rather than calling `star_sim.preprocess()`. Once you plug in your real `"preprocesada"` function, switch your training and baseline scripts to use it consistently.

---

### Calibration Checklist & Gate 1 Test

1. Edit `Telescope` and `Ranges` in [`Fourier-optics simulator/star_sim.py`](<../Fourier-optics%20simulator/star_sim.py>) with your Day 2 telescope, camera, and range numbers (keep `side <= grid`, or increase `grid` to `1024` if needed — note `grid=1024` is ~4× slower, so test runtime on 100 images first).
2. Replace `preprocess()` in `star_sim.py` with your real preprocessing function.
3. Run the **Gate 1 asymmetry check** below from inside `Fourier-optics simulator/`:

```python
# gate1_check.py — verify intra- vs. extra-focal asymmetry for YOUR Telescope config
import numpy as np
import matplotlib.pyplot as plt
from star_sim import StarSimulator, Telescope

tel = Telescope()
sim = StarSimulator(tel=tel)
print(f"f/{tel.fnum:.2f}, dz_per_wave = {tel.dz_per_wave():.1f} um, zoom = {sim.zoom:.2f}")

# Noise-free pair at +/- 8 waves (adjust W to ~70% of your max defocus_waves)
W = 8.0
base = dict(seeing_fwhm_px=8.0, moffat_beta=3.5, d_over_r0=0.01, low_order=0.0, shift=0.0)
intra = sim.batch(1, fixed={**base, "defocus_waves": -W}, clean=True)[0].numpy()
extra = sim.batch(1, fixed={**base, "defocus_waves": +W}, clean=True)[0].numpy()

diff_l1 = np.abs(intra - extra).sum() / intra.sum()
print(f"Normalized L1 intra/extra difference at +/-{W} waves: {diff_l1:.3f}")

y, x = np.mgrid[:tel.cutout, :tel.cutout] - (tel.cutout - 1) / 2
r = np.hypot(x, y).astype(int)
prof_in  = [intra[r == k].mean() for k in range(tel.cutout // 2)]
prof_out = [extra[r == k].mean() for k in range(tel.cutout // 2)]

fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
ax[0].imshow(intra, cmap="gray", origin="lower"); ax[0].set_title(f"Inside focus (-{W} waves)")
ax[1].imshow(extra, cmap="gray", origin="lower"); ax[1].set_title(f"Outside focus (+{W} waves)")
ax[2].plot(prof_in, label=f"intra (-{W})"); ax[2].plot(prof_out, label=f"extra (+{W})")
ax[2].set_xlabel("radius (px)"); ax[2].set_title(f"Radial profile (L1 diff = {diff_l1:.3f})"); ax[2].legend()
plt.tight_layout(); plt.savefig("gate1_asymmetry.png", dpi=120)
```

**Gate 1 pass criterion:** `intra` and `extra` show a clear, visible radial-profile difference (`gate1_asymmetry.png`), just like the bottom-right panel of `sim_vs_real_v1.png`.

> [!CAUTION]
> **Gate 1 Fallback (if your optics turn out completely symmetric):** If you own a well-corrected unobstructed refractor (`obstruction = 0`, `spherical_waves ≈ 0`), `intra` and `extra` will overlap almost completely (`diff_l1 ≈ 0`). Do **not** waste Weeks 2–4 trying to train a signed CNN on symmetric PSFs. Instead, apply the Risk Fallback from the [Problem Statement & Plan](<../Single-Image%20Telescope%20Autofocus%20—%20Problem%20Statement%20&%20Plan.md>):
> 1. Measure whether real frames actually show slight residual spherical aberration, astigmatism, or coma from your optical train and model that; **or**
> 2. Pivot the single-image task to predicting **magnitude $|\Delta z|$** cleanly and pair it with a one-step focuser probe move to resolve the sign.

---

## Day 4 — Regenerate & Audit Your Calibrated Dataset

Once `Telescope`, `Ranges`, and `preprocess()` pass Gate 1, generate your calibrated 20,000-image dataset (`~7 min` on 2 CPU cores).

> [!CAUTION]
> **Beware the stale-shards trap in [`make_dataset_v1.py`](<../Fourier-optics%20simulator/make_dataset_v1.py>)!** Line 19 checks `if os.path.exists(path): continue` so interrupted runs can resume. If you edit `star_sim.py` and re-run `make_dataset_v1.py` without removing the old `shards/` directory first, it will silently reuse old shards from the previous telescope! Always delete or archive `shards/` before rebuilding after any config change.

```bash
cd "Fourier-optics simulator"
rm -rf shards                    # PowerShell: Remove-Item -Recurse -Force shards -ErrorAction SilentlyContinue
python make_dataset_v1.py 20000 defocus_stars_custom.npz
python plot_samples_v1.py defocus_stars_custom.npz samples_custom.png
```

Inspect `samples_custom.png` carefully (do **not** skip visual audit — Phase 6 caught clipped donuts and excessive turbulence speckle on the first pass by looking at the PNG):
- Do the largest positive and negative `dz` donuts stay inside the `64×64` frame even with jitter?
- Does the speckle / seeing smoothness resemble your real frame?
- Run a quick sanity check on `val_idx` (copy `train_check_v1.py` to a version that logs `val_idx` and uses your `preprocess()`), keeping `test_idx` untouched.

---

## Day 5 — Build the Classical HFR Baseline (Gate 2)

To prove in Week 3 that a CNN is actually worthwhile, you first need a rock-solid classical benchmark on the **exact same dataset and $|\Delta z|$ bins**. That is **Gate 2**.

Recall how **Half-Flux Radius (HFR)** works:
1. Subtract the local sky + bias background from each `64×64` star cut-out (e.g., median of the outer 4-pixel border) and clip negative noise fluctuations at `0`.
2. Compute the flux-weighted centroid $(x_c, y_c)$ inside the cut-out.
3. Compute every pixel's distance $r_i$ from $(x_c, y_c)$, sort pixels by $r_i$, and find the radius where cumulative flux crosses `50%` of the total positive flux (`total_flux / 2`).
4. Fit a monotonic mapping `HFR -> |dz|` using **`train_idx` ONLY** (for example, `np.interp` on quantile-binned `(HFR_train, |dz_train|)` averages or a low-degree polynomial/isotonic fit).
5. Evaluate $|\hat{d}_z| - |\Delta z|$ MAE per $|\Delta z|$ bin on `val_idx` and `test_idx`.

Here is a clean starter script (`hfr_baseline.py`, save inside `Fourier-optics simulator/`):

```python
# hfr_baseline.py — Classical Half-Flux Radius (HFR) baseline for Gate 2
import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "defocus_stars_v1.npz"
z = np.load(path)
adu = z["adu"].astype(np.float32)
dz = z["dz_um"].astype(np.float32)
tr, va, te = z["train_idx"], z["val_idx"], z["test_idx"]

def compute_hfr(stack):
    n, h, w = stack.shape
    # 4-pixel border background estimate (matching train_check_v1.py)
    border = np.concatenate([
        stack[:, :4].reshape(n, -1), stack[:, -4:].reshape(n, -1),
        stack[:, :, :4].reshape(n, -1), stack[:, :, -4:].reshape(n, -1),
    ], axis=1)
    bg = np.median(border, axis=1)[:, None, None]
    flux = np.clip(stack - bg, 0.0, None)

    yy, xx = np.mgrid[:h, :w]
    tot = flux.sum(axis=(1, 2)) + 1e-8
    xc = (flux * xx[None]).sum(axis=(1, 2)) / tot
    yc = (flux * yy[None]).sum(axis=(1, 2)) / tot

    hfr = np.empty(n, dtype=np.float32)
    for i in range(n):
        r = np.hypot(xx - xc[i], yy - yc[i]).ravel()
        f = flux[i].ravel()
        order = np.argsort(r)
        r_s, f_s = r[order], f[order]
        cum = np.cumsum(f_s)
        half = 0.5 * cum[-1]
        idx = int(np.searchsorted(cum, half))
        hfr[i] = r_s[min(idx, len(r_s) - 1)]
    return hfr

hfr = compute_hfr(adu)

# Fit monotonic HFR -> |dz| calibration curve on TRAIN split only (20 quantile bins)
qs = np.linspace(0, 100, 21)
edges = np.percentile(hfr[tr], qs)
bin_hfr, bin_abs_dz = [], []
for a, b in zip(edges[:-1], edges[1:]):
    m = (hfr[tr] >= a) & (hfr[tr] <= b)
    if m.any():
        bin_hfr.append(hfr[tr][m].mean())
        bin_abs_dz.append(np.abs(dz[tr][m]).mean())
order = np.argsort(bin_hfr)
cal_hfr = np.array(bin_hfr)[order]
cal_dz  = np.array(bin_abs_dz)[order]

for name, split in [("VAL", va), ("TEST", te)]:
    pred_abs = np.interp(hfr[split], cal_hfr, cal_dz)
    true_abs = np.abs(dz[split])
    print(f"\n=== HFR Baseline ({name}) — overall |dz| MAE: {np.abs(pred_abs - true_abs).mean():.1f} µm ===")
    print("|dz| bin (µm)   n    HFR mean   |dz| MAE (µm)")
    for a, b in zip([0, 200, 400, 800, 1200], [200, 400, 800, 1200, 1600]):
        m = (true_abs >= a) & (true_abs < b)
        if m.any():
            print(f"{a:>5}-{b:<5}   {m.sum():>5}   {hfr[split][m].mean():>7.2f}     {np.abs(pred_abs - true_abs)[m].mean():>9.1f}")
```

Notice what this gives you immediately:
- An honest, reproducible **Gate 2** error table across every $|\Delta z|$ bin.
- A benchmark that uses **zero** signed information (HFR alone cannot tell inside from outside focus, so any guess at sign is ~50% random).
- Notice also a key diagnostic question: at very low SNR (`ring_snr ≈ 3`) or near focus (`|dz| < 200 µm`), noisy background pixels clipped above zero can inflate HFR! Observing *where* HFR breaks down tells you why Week 3's CNN should beat it.

---

## Day 6 — Real-Sky Preparation: Focuser Sweep Protocol or MILAN FITS

Synthetic accuracy only matters if it transfers to real sky frames. Use Day 6 to prepare real evaluation data for Week 5:

### Option A (Preferred, if skies are clear): Capture a Real FITS Focuser Sweep
1. Point at a field containing at least one **isolated, medium-bright star** (avoid saturated cores above ~70% of your sensor's full well and avoid overlapping companions inside a `64×64` box).
2. Using **NINA**, **Ekos**, or **ASCOM**, sweep the focuser across **15 to 25 evenly spaced steps** from well inside focus to well outside focus (matching your simulated `±dz` span as closely as practical).
3. Save **raw 16-bit FITS** files (no stretching, no debayering if mono, or extract a single channel consistently if OSC) and verify that `FOCPOS` (or focuser step position) is recorded in each FITS header.
4. Fit a classical V-curve (hyperbola or parabola) to `HFR(FOCPOS)` to estimate best focus $p^*$, and compute each frame's ground-truth label:
   $$\Delta z = \texttt{FOCPOS} - p^*$$
5. Bonus insight: visually compare frames at equal $|\Delta z|$ on both sides of $p^*$ — this reveals your telescope's **true** spherical aberration direction and strength!

### Option B (Cloudy weather): Test Your Pipeline on MILAN Raw FITS
Download a sample raw Stellina FITS file from the [MILAN Sky Survey on Zenodo](https://zenodo.org/records/7304607), extract a `64×64` star cut-out with `astropy.io.fits`, run your `preprocess()` and `compute_hfr()` functions on it, and inspect both the raw ADU values and preprocessed image side by side with a synthetic sample from `make_dataset_v1.py`.

---

## Day 7 — Pre-Class Deliverable Checklist

Bring these five items to next week's class so we can clear **Gate 1 & Gate 2** in the first 15 minutes and jump straight into **Week 3 (CNN vs. HFR)**:

1. **Completed Week-1 decisions table** (optics/camera specs, $\Delta z$ unit, metric targets, sweep plan).
2. **Updated `Telescope`, `Ranges`, and real `preprocess()`** in [`Fourier-optics simulator/star_sim.py`](<../Fourier-optics%20simulator/star_sim.py>).
3. **Gate 1 evidence (`gate1_asymmetry.png` + `samples_custom.png`)** confirming intra/extra asymmetry is visible and donuts fit cleanly inside `64×64`.
4. **Gate 2 baseline table** from `hfr_baseline.py` on `val_idx` (plus initial 4-epoch CNN numbers if you ran `train_check_v1.py` on your custom dataset).
5. **Real-data status:** either your raw FITS sweep (`FOCPOS` + V-curve best focus $p^*$) or one inspected real FITS star cut-out showing raw ADU stats (`sky_med`, `bg_std`, `ring_snr`).

---

## Troubleshooting & Common Pitfalls

| Symptom | Likely Cause | Fix |
|---|---|---|
| `zipfile.BadZipFile: File is not a zip file` on `np.load(...)` | `.npz` is still a 130-byte Git LFS pointer | Run `git lfs install && git lfs pull` and check byte sizes (`~60 MB` / `~109 MB`) |
| Rebuilt `.npz` still shows old telescope's stars | `make_dataset_v1.py` reused cached files in `shards/` | Delete `Fourier-optics simulator/shards/` before regenerating |
| Shape/pool error or truncated PSF in `StarSimulator.batch()` | `side = ceil(cutout * zoom)` exceeded `grid = 512` | Print `side`; if `> 512`, pass a larger `grid` (e.g., `1024`) to `StarSimulator` |
| Synthetic CNN scores high, real star prediction fails | `train_check_v1.py` used border normalization while real image used different `"preprocesada"` scaling | Replace `preprocess()` in `star_sim.py` and call it identically on synthetic and real cut-outs |
| Unsure why `Benchmark v0/` gives `98.2%` sign easily | Invented ring-width shortcut `w = 1.5 + 0.5·d` | Never train/evaluate autofocus models on `Benchmark v0/`; use `Fourier-optics simulator/` |
| Overfitting design choices to `test_idx` | Tuning hyperparameters directly against `test_idx` | Evaluate design iterations on `val_idx` (`2,000` samples); freeze `test_idx` until final checks |

---

## Summary & Bridge to Next Class

This week shifts the project from a generic `200 mm f/5` placeholder to **your** optical reality: you have checked the physics by hand (`8 N² λ`), replaced the placeholder preprocessing and optics in `star_sim.py`, verified that intra- and extra-focal PSFs look genuinely different (Gate 1), rebuilt a clean 20,000-star training/validation/test dataset, and measured the classical HFR baseline per $|\Delta z|$ bin (Gate 2).

**Next class (Week 3 — First CNN vs. HFR Baseline, Gate 3):** Using your calibrated dataset and Gate 2 HFR table, we will train a small CNN first on blur magnitude $|\Delta z|$ to verify it beats the HFR baseline on `val_idx`, diagnose where HFR struggles (low SNR and near-focus), and then move to signed $\Delta z$ prediction and robustness augmentations in Week 4.

---

## Self-Check Quiz (5 Questions)

1. **Why is an unobstructed, aberration-free refractor (`obstruction = 0`, `spherical_waves = 0`, no coma/astigmatism) a critical-path blocker for single-image signed autofocus?**
   - **A.** Because its FFT PSF cannot fit inside a `64×64` pixel array.
   - **B.** Because pure symmetric defocus produces identical PSFs inside and outside focus, making $\operatorname{sign}(\Delta z)$ mathematically unobservable from one frame.
   - **C.** Because without a central obstruction, HFR becomes negative inside focus.

2. **Suppose you keep the same sensor (`pixel_um = 3.76`) and wavelength (`λ = 0.55 µm`), but switch from an `f/5` telescope (`dz_per_wave = 110 µm`) to an `f/10` telescope (`focal_mm = 2000`, `aperture_mm = 200`). Using $8 N^2 \lambda$, what is the new `dz_per_wave`?**
   - **A.** `220 µm/wave` (doubles linearly with focal ratio $N$).
   - **B.** `440 µm/wave` (quadruples because depth of focus scales with $N^2$).
   - **C.** `55 µm/wave` (halves because the light cone is narrower).

3. **Why must you NOT claim that the autofocus sign task works based on the `98.2%` sign accuracy in `Benchmark v0/`?**
   - **A.** Because `Benchmark v0/` has only 2,000 training images and no validation split.
   - **B.** Because `Benchmark v0/` uses `uint16` photon counts instead of `float32` ADU.
   - **C.** Because `Benchmark v0/` encodes the sign with an artificial ring-width formula (`w = 1.5 + 0.5·d`), not physical optical aberrations.

4. **You change `obstruction = 0.30` to `0.38` in `star_sim.py` and immediately run `python make_dataset_v1.py 20000 my_tel.npz` while an old `shards/` folder is still present. What happens?**
   - **A.** `make_dataset_v1.py` detects the `Telescope` change via `config` JSON and overwrites all shards cleanly.
   - **B.** `make_dataset_v1.py` skips every existing `shards/shard_XX.npz` and merges the *old* `0.30`-obstruction images into `my_tel.npz`.
   - **C.** Python raises a `FileExistsError` and refuses to run until `shards/` is deleted.

5. **Which combination of practices prevents data leakage and sim-to-real mismatch when iterating on your model this week?**
   - **A.** Tune hyperparameters and HFR bin mappings on `train_idx` / `val_idx` only (keeping `test_idx` frozen), and apply the exact same `preprocess()` function to synthetic and real frames.
   - **B.** Evaluate every tweak on `test_idx` to get the cleanest estimate, and normalize synthetic frames with `4-px` border stats while normalizing real frames with your camera software's stretch.
   - **C.** Train on `Benchmark v0/` first so the CNN learns the sign quickly, then fine-tune on `test_idx`.

---

## Answer Key

1. **Correct: B.** Under pure symmetric defocus, reversing the sign of defocus flips the phase $\exp(i 2\pi W)$ to its complex conjugate, leaving $|\text{FFT}(\text{pupil})|^2$ identical on both sides of focus. **A** is wrong because cut-out fit depends on focal ratio and $\Delta z$, not symmetry. **C** is wrong because HFR is a non-negative radius.
2. **Correct: B.** Since $\Delta z_{\text{per wave}} = 8 N^2 \lambda$, doubling $N$ from `5` to `10` multiplies $N^2$ by $4$, giving $8 \times 100 \times 0.55 = 440\ \mu\text{m/wave}$ (four times `110 µm`). **A** and **C** assume linear or inverse scaling instead of $N^2$.
3. **Correct: C.** As documented in the [Dataset Card](<../Dataset%20Card%20—%20Toy%20Defocused-Star%20Set%20%28benchmark%20v0%29.md>), `Benchmark v0/` was created solely to time PyTorch on CPU and invents `w = 1.5 + 0.5·d`, which does not reflect real telescope optics. **A** is wrong (`Benchmark v0/` has `18,000` training images), and **B** is wrong because data storage dtype (`uint16` scaled by `200`) is not the reason sign accuracy is invalid.
4. **Correct: B.** Line 19 of [`make_dataset_v1.py`](<../Fourier-optics%20simulator/make_dataset_v1.py>) executes `if os.path.exists(path): continue` so crashed runs can resume, without checking whether `Telescope` or `Ranges` changed. **A** and **C** describe behaviors the script does not implement.
5. **Correct: A.** Fitting calibration curves on `train_idx`, iterating against `val_idx`, reserving `test_idx` for final evaluation, and applying one shared `preprocess()` function across synthetic and real images keeps evaluation honest and eliminates preprocessing shift. **B** causes both `test_idx` leakage and sim-to-real input mismatch; **C** leaks `test_idx` and relies on the non-physical toy shortcut.
