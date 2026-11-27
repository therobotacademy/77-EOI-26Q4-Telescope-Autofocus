# 01 — What Is the HFR Baseline?

> **Question:** *What is the HFR baseline?*

The **HFR baseline** is the project's **classical (non-deep-learning) autofocus benchmark** based on **Half-Flux Radius (HFR)**. Given a single `64×64` defocused star cut-out, it measures how wide the star's light distribution is and maps that radius to a predicted defocus magnitude $|\Delta z|$—establishing the error floor that the Week 3 CNN must beat.

---

## 1. Definition of Half-Flux Radius (HFR)

For an isolated star image after sky/bias subtraction, the **Half-Flux Radius (HFR)** is the radial distance $r$ from the star's flux-weighted centroid $(x_c, y_c)$ that encloses **50% of the total positive star flux**:

$$
\sum_{r_i \le \text{HFR}} F_i \;\ge\; 0.5 \sum_{\text{all } i} F_i
$$

In classical astronomical autofocus, a telescope moves through focus across multiple frames to trace a **V-curve** of $\text{HFR}$ versus focuser position (`FOCPOS`). Near best focus ($p^*$), HFR reaches its minimum; away from focus, the defocused blur disk expands roughly linearly with $|\Delta z|$, increasing HFR on both sides.

---

## 2. Role in the 6-Week Gated Roadmap

The HFR baseline serves three distinct roles across the project ([`README.md`](../README.md), [`Single-Image Telescope Autofocus — Problem Statement & Plan.md`](<../Single-Image%20Telescope%20Autofocus%20—%20Problem%20Statement%20&%20Plan.md>)):

| Stage | How HFR Is Used | Gate / Criterion |
|---|---|---|
| **Week 2 (Gate 2)** | Compute HFR on synthetic star cut-outs, calibrate `HFR -> |Δz|` on `train_idx`, and measure $|\Delta z|$ MAE per $|\Delta z|$ bin. | HFR error measured per $|\Delta z|$ bin on the synthetic test/validation set. |
| **Week 3 (Gate 3)** | Benchmark the first blur-size CNN directly against HFR on the **exact same dataset and $|\Delta z|$ bins**. | Small 4-block CNN beats the HFR baseline MAE (`Beat HFR baseline`). |
| **Week 5 (Real Sky)** | 1. Fit a V-curve (hyperbola/parabola) to `HFR(FOCPOS)` on a real focuser sweep to locate $p^*$ and label $\Delta z = \texttt{FOCPOS} - p^*$.<br>2. Evaluate blur-size correlation against classical HFR on unlabeled [MILAN Sky Survey](https://zenodo.org/records/7304607) FITS frames. | Real-vs-synthetic error gap and MILAN HFR correlation reported. |

---

## 3. How the HFR Baseline Is Computed

Following the Gate 2 specification in [`STUDENT_WEEK1_GUIDE.md`](STUDENT_WEEK1_GUIDE.md) (Day 5), the baseline runs in five steps:

1. **Estimate and subtract local background:** Take the median of the outer 4-pixel border of each `64×64` cut-out (matching the normalization border in [`train_check_v1.py`](<../Fourier-optics%20simulator/train_check_v1.py>)) and clip negative noise residuals at `0`.
2. **Compute the flux-weighted centroid $(x_c, y_c)$:** Weight pixel coordinates by the background-subtracted positive flux.
3. **Compute HFR:** Calculate each pixel's distance $r_i = \sqrt{(x_i - x_c)^2 + (y_i - y_c)^2}$, sort pixels by $r_i$, and find the radius where cumulative flux crosses `50%` of the total positive flux.
4. **Calibrate `HFR -> |Δz|` on `train_idx` only:** Fit a monotonic mapping (e.g., 20 quantile-binned `(mean HFR, mean |Δz|)` points interpolated via `np.interp`, or an isotonic/low-degree polynomial fit) strictly on `train_idx` to avoid leakage into `val_idx` or `test_idx`.
5. **Evaluate per $|\Delta z|$ bin:** Report overall $|\hat{\Delta z}| - |\Delta z|$ MAE and per-bin MAE (`0–200`, `200–400`, `400–800`, `800–1200`, `1200–1600 µm`) on `val_idx` and `test_idx`.

---

## 4. Why HFR Alone Is Not Enough

HFR is a useful classical yardstick, but it cannot solve single-image autofocus on its own:

- **Sign-blind ($\approx 50\%$ sign accuracy):** HFR is a purely scalar measure of blur *radius* ($|\Delta z|$). It discards ring sharpness, central-hole depth, and radial profile asymmetry, so from a single frame it cannot distinguish inside focus ($\Delta z < 0$) from outside focus ($\Delta z > 0$). That is why classical autofocus requires a multi-frame sweep.
- **Sensitivity to low SNR:** Clipping noisy sky pixels above zero at low signal-to-noise ratios (e.g., `ring_snr ≈ 3` in Dataset v1) adds spurious outer flux and inflates HFR.
- **Near-focus floor (`|dz| < 200 µm`):** Close to best focus, atmospheric seeing (`FWHM 6–12 px` in Dataset v1) dominates over geometric defocus, flattening the `HFR` vs. `|Δz|` curve.

---

## 5. Current Status in This Repository

- **Status:** **Pending execution.** Physical Dataset v1 ([`defocus_stars_v1.npz`](<../Fourier-optics%20simulator/defocus_stars_v1.npz>)) is built and the 4-epoch CNN sanity check ([`train_check_v1.py`](<../Fourier-optics%20simulator/train_check_v1.py>)) has been run (`148 µm` test MAE), but **no Gate 2 HFR evaluation script has been executed or committed yet**, so no measured HFR MAE numbers exist in the repository yet.
- **Starter implementation:** A ready-to-run `hfr_baseline.py` starter script is documented in Day 5 of [`STUDENT_WEEK1_GUIDE.md`](STUDENT_WEEK1_GUIDE.md).
