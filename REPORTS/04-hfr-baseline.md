# 04 · HFR baseline (week 2)

Run and written 2026-11-27.

## Objective and gate

Set the classical reference for blur size. **Gate:** HFR error measured per |Δz| bin on the synthetic test set.

## Method

- [`hfr_baseline.py`](../Fourier-optics%20simulator/hfr_baseline.py), numpy only: `python hfr_baseline.py defocus_stars_v1.npz`.
- Per image: subtract the border median, find the flux centroid, take the radius enclosing half the flux.
- HFR → |Δz|: monotone curve of median |Δz| per HFR bin (80 bins), fitted on 16,000 train images.
- Centroid threshold (1–3σ) and aperture (26–46 px) chosen on the validation split: 2σ, 38 px.
- Depth of focus (DoF): quarter-wave tolerance ±2λN² = ±27.5 µm.

## RESULTS

Test split, 2,000 images: **MAE of |Δz| = 64.9 µm (2.4 DoF)**, median error 53.6 µm. A constant guess gives 302.3 µm.

| \|Δz\| (µm) | n | MAE (µm) | MAE (DoF) | Median error (µm) | Bias (µm) |
|---|---|---|---|---|---|
| 0–200 | 357 | 84 | 3.0 | 72 | +72 |
| 200–400 | 308 | 77 | 2.8 | 71 | −25 |
| 400–800 | 664 | 65 | 2.3 | 55 | −10 |
| 800–1200 | 657 | 49 | 1.8 | 39 | −8 |
| 1200–1600 | 14 | 75 | 2.7 | 59 | −75 |

Validation MAE ranged from 66.3 to 68.9 µm over the settings tried.

## Discussion

- **Gate passed.**
- The result is insensitive to the measurement settings.
- Near focus the estimate is biased high (+72 µm): seeing sets a floor on the star's size, so small defocus is hard to separate.
- The last bin holds only 14 images (the data stop at 1,210 µm).
- The curve is fitted on thousands of labelled stars from the same simulator. A real V-curve calibration would have far less, so this is a demanding reference.
- HFR gives no sign, so it can only be compared on |Δz|.

## Next steps

- Compare the CNN with this baseline on |Δz|, same test split (report 05).
- Repeat the baseline after any recalibration of the simulator.
- Confirm the depth-of-focus definition and the near-focus zone.
