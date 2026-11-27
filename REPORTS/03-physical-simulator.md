# 03 · Physical simulator and dataset v1 (week 1)

Work dated 2026-11-18. Written 2026-11-27. Calibration and training figures are as logged then and were not re-run; dataset shapes and ranges were checked on 2026-11-27.

## Objective and gate

Build synthetic stars that resemble a real raw frame. **Gate:** the sign is visible, meaning intra- and extra-focal images differ at equal |Δz|.

## Method

- [`star_sim.py`](../Fourier-optics%20simulator/star_sim.py): obstructed pupil, defocus, 0.4 waves spherical aberration, random astigmatism and coma, Kolmogorov turbulence, Moffat seeing, sensor noise. Details in [README_v1](../Fourier-optics%20simulator/README_v1.md).
- Assumed optics: 200 mm f/5, 30% obstruction, 3.76 µm pixels.
- Calibration: radial profile fitted to one real frame read from a screenshot.
- Build: `python make_dataset_v1.py 20000 defocus_stars_v1.npz`.
- Check: `python train_check_v1.py defocus_stars_v1.npz 4`.

## RESULTS

**Dataset v1:** 20,000 images of 64×64 (uint16 ADU), Δz within ±1,210 µm, split 16,000 / 2,000 / 2,000.

**Calibration:** best fit at 9 waves of defocus, seeing FWHM 10 px, Moffat β 4, D/r0 6; reported radial-profile RMS error 0.04.

![Simulator against the real frame](../Fourier-optics%20simulator/sim_vs_real_v1.png)

![Dataset v1 samples](../Fourier-optics%20simulator/samples_v1.png)

**CNN check** (4 epochs, signed Δz): MAE 148 µm, sign accuracy 88.9%.

| \|Δz\| (µm) | Sign accuracy |
|---|---|
| 0–200 | 58% |
| 200–400 | 90% |
| 400–800 | 95% |
| above 800 | 98% |

## Discussion

- **Gate passed in simulation only.** The noise-free intra and extra profiles differ clearly, and the CNN reads the sign well above chance away from focus.
- The sign cue comes from an assumed 0.4 waves of spherical aberration. Its real size is unmeasured.
- In the calibration figure the synthetic profiles peak near 16 px, while the real one has a broad top at about 11–16 px and a higher inner shoulder. The fitted D/r0 of 6 also lies outside the dataset's 2–5 range. The figure should be checked against the quoted RMS.
- Calibration rests on one screenshot, not raw FITS. There are no spider vanes, colour, saturation, hot pixels or neighbouring stars, and `preprocess()` is a placeholder.

## Next steps

- Get the real telescope and camera values and a raw FITS frame; recalibrate.
- Capture a focuser sweep to measure the spherical aberration.
- Measure the classical baseline on this dataset (report [04](04-hfr-baseline.md)).
