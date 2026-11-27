# 06 · Signed defocus and robustness (week 4)

Run and written 2026-11-27.

## Objective and gate

Predict the signed focus error Δz. **Gate:** the sign target (proposed ≥ 95%) is met outside the near-focus zone, across seeing, centre offset and SNR.

The near-focus zone has not been agreed, so two limits are reported: |Δz| below 200 µm and below 400 µm.

## Method

- [`signed_robustness.py`](../Fourier-optics%20simulator/signed_robustness.py), conda env `xgnn` (RTX 2060): `python signed_robustness.py defocus_stars_v1.npz 300 30 3`.
- Signed 4-block CNN of report [05](05-cnn-blur-size.md) (annexes A and C), same normalisation, L1 loss, Adam 1e-3, batch 128.
- Training stops after 30 epochs without a lower validation MAE (300 at most). The best-validation weights are scored once on the 2,000 test images.
- Three seeds. Tables show the mean of the three models' scores.
- Weights and test predictions are saved in `Fourier-optics simulator/weights/`.

## RESULTS

### Training

| Seed | Best epoch | Epochs run | Val MAE (µm) | Test signed MAE (µm) | Test sign accuracy |
|---|---|---|---|---|---|
| 0 | 78 | 108 | 60.9 | 63.3 | 93.2% |
| 1 | 70 | 100 | 69.5 | 71.2 | 92.8% |
| 2 | 54 | 84 | 62.8 | 64.4 | 92.6% |

### By |Δz| bin

| \|Δz\| (µm) | n | Sign accuracy | Signed MAE (µm) | \|Δz\| MAE (µm) |
|---|---|---|---|---|
| 0–100 | 183 | 62.7% | 75.7 | 53.2 |
| 100–200 | 174 | 74.3% | 102.1 | 70.5 |
| 200–300 | 153 | 92.4% | 85.2 | 73.0 |
| 300–400 | 155 | 97.2% | 70.6 | 63.3 |
| 400–600 | 314 | 97.6% | 66.2 | 55.4 |
| 600–800 | 350 | 99.1% | 54.6 | 48.3 |
| 800–1210 | 671 | 99.5% | 55.4 | 50.7 |

### Outside the near-focus zone

| Images | n | Sign accuracy | Per seed | Signed MAE (µm) |
|---|---|---|---|---|
| All | 2000 | 92.8% | 93.2 / 92.8 / 92.6% | 66.3 |
| \|Δz\| ≥ 200 µm | 1643 | **98.2%** | 98.4 / 97.9 / 98.2% | 61.5 |
| \|Δz\| ≥ 400 µm | 1335 | **98.9%** | 99.3 / 98.7 / 98.8% | 57.7 |

At |Δz| ≥ 200 µm, sign accuracy is 97.8% inside focus (821 images) and 98.5% outside (822).

### Robustness

| Condition | n (≥ 200) | Sign acc. (≥ 200) | MAE (≥ 200) | n (≥ 400) | Sign acc. (≥ 400) | MAE (≥ 400) |
|---|---|---|---|---|---|---|
| Seeing FWHM 6–8 px | 558 | 98.2% | 61.6 | 455 | 99.0% | 57.0 |
| Seeing FWHM 8–10 px | 571 | 98.5% | 57.6 | 458 | 99.1% | 55.1 |
| Seeing FWHM 10–12 px | 514 | 97.7% | 65.8 | 422 | 98.7% | 61.3 |
| Centre offset 0–1.5 px | 327 | 97.7% | 65.4 | 245 | 98.4% | 64.3 |
| Centre offset 1.5–2.5 px | 570 | 98.3% | 62.8 | 478 | 99.2% | 58.9 |
| Centre offset 2.5–4.3 px | 746 | 98.3% | 58.7 | 612 | 99.0% | 54.1 |
| Ring SNR 3–6 | 553 | 97.9% | 67.2 | 465 | 98.9% | 62.8 |
| Ring SNR 6–9 | 527 | 98.0% | 57.9 | 431 | 98.6% | 54.8 |
| Ring SNR 9–12 | 563 | 98.5% | 59.2 | 439 | 99.3% | 55.2 |
| D/r0 2–3 | 568 | 99.2% | 40.8 | 464 | 99.7% | 37.0 |
| D/r0 3–4 | 530 | 97.8% | 63.0 | 427 | 99.1% | 56.2 |
| D/r0 4–5 | 545 | 97.4% | 81.6 | 444 | 97.9% | 80.9 |

## Discussion

- **Gate passed in simulation.** Sign accuracy is above 95% in every condition tested, for both limits.
- **The near-focus zone matters.** 200 µm is about 7 depths of focus (±27.5 µm), which is wide. The 200–300 µm bin (92.4%) misses 95%, and below 100 µm the sign is little better than guessing. An autofocus loop must cope with this.
- **Turbulence has the largest effect** of the conditions tested: MAE doubles from D/r0 2–3 to 4–5. Seeing, centre offset and SNR change little within the simulated ranges.
- Longer training added little to the sign: 92.6–93.2% overall, against 92.3–92.8% at 40 epochs. Mean signed MAE went from 69.6 to 66.3 µm. Validation MAE fluctuates from epoch to epoch, and seed 1 stayed worse.
- On |Δz| the signed model averages about 55.7 µm (from the bin table), against 64.9 µm for HFR and 46.6 µm for the size model of report 05.
- Each condition is split on its own; combined worst cases (strong turbulence with low SNR) were not examined.
- The sign cue comes from an **assumed** 0.4 waves of spherical aberration. These results would change if the real optics differ, and nothing has been tested on real stars.
- The test split was also used for reports 04 and 05. Nothing in this run was selected on it.

## Next steps

- Agree the near-focus limit and the sign target.
- Simulated closed-loop autofocus, including behaviour near focus (report [08](README.md)).
- Real images (report [07](README.md)): MILAN for blur size, a focuser sweep for the sign.
- Test sensitivity to the spherical aberration value, which can be done in simulation before real data arrive.
- Consider combining the size model's magnitude with the signed model's sign.
