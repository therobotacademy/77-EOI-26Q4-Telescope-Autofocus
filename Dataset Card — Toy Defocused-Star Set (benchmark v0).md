# Dataset Card — Toy Defocused-Star Set (benchmark v0)

Nov 18, 2026 · @Bernardo

## Summary

This set is a toy, made only to time CNN training in the sandbox. It is not the project's star simulator and must not be used to judge the autofocus model. It holds 20,000 single-channel 64×64 images of a ring-shaped "defocused star", each labelled with a signed defocus value d in \[−1, 1\].

- **Made with:** `bench.py`, PyTorch 2.15 (CPU), seed 0, on 2026-11-18.
- **Storage:** generated in memory each run (about 330 MB as float32), never written to disk.
- **Key caveat:** the sign of d is encoded by a rule chosen for the benchmark, not by optics. Its sign scores are therefore meaningless for the real task (see Limitations).

## Generation

Each image is a Gaussian ring whose radius grows with |d| and whose width grows with d, plus a flat background and Poisson noise. No optics are simulated.

1. Draw the label d uniformly from \[−1, 1\].
2. For every pixel, take its distance r from the point (32, 32).
3. Set the ring radius R = 2 + 12·|d| pixels and the ring width w = 1.5 + 0.5·d pixels.
4. Compute the clean intensity, then add background and shot noise:

$$
I(r) = \exp\!\left(-\frac{(r - R)^2}{2w^2}\right), \qquad \text{pixel} = \frac{\mathrm{Poisson}(200\,I + 10)}{200}
$$

The peak signal is 200 counts on a background of 10 counts. Step 3's width rule is what makes the sign learnable: rings are thinner inside focus (w = 1.0 at d = −1) than outside (w = 2.0 at d = +1).

## Structure

The set is two in-memory tensors, an image stack X and a label vector Y, split by index into 18,000 training and 2,000 test images. The values below were measured on the seed-0 set.

| Field | Shape and type | Values |
| --- | --- | --- |
| X (image) | 20,000 × 1 × 64 × 64, float32 | 0.0 to 1.40, mean 0.096; per-image peak 1.06 to 1.40 |
| Y (label d) | 20,000, float32 | −1.000 to +1.000, mean +0.003; 50.3% positive |

| Split | Images | Share with d > 0 |
| --- | --- | --- |
| Train (index 0–17,999) | 18,000 | 50.3% |
| Test (index 18,000–19,999) | 2,000 | 50.1% |

There is no validation split, and no metadata is stored per image. d has no physical unit; it is a normalised defocus.

## Intended use and limitations

Use it only to time training, test a training loop end to end, or show the pipeline in class. Do not report its scores as autofocus results.

**Out of scope:** training or evaluating the real model, comparing architectures for the real task, and any claim about whether the sign is learnable.

**Known limitations:**

- **Sign shortcut.** The width rule w = 1.5 + 0.5·d was invented for the benchmark. Real optics break the symmetry in other ways (spherical aberration, central obstruction), so sign accuracy here says nothing about real stars.
- **No optics.** No pupil, aperture or diffraction pattern, and no spider vanes from the secondary mirror's supports.
- **No atmosphere or sensor effects.** No seeing, read noise, saturation, hot pixels or sky gradient.
- **Ideal framing.** One star per image, always at (32, 32), at the same brightness. Real cut-outs are off-centre and vary in flux.
- **No metadata.** Only d is kept, so no analysis by noise level or brightness is possible.

## Baseline and reproduction

A small 4-block CNN (32, 64, 128, 128 channels, MSE loss, Adam at 1e-3, batch 128) reached a test MAE of 0.012 and 98.2% sign accuracy after 3 epochs, in 172 s on the sandbox. Given the shortcut, these numbers only show that the loop trains.

| Epoch | Test MAE (d) | Sign accuracy | Elapsed |
| --- | --- | --- | --- |
| 1 | 0.021 | 97.8% | 57 s |
| 2 | 0.013 | 98.3% | 114 s |
| 3 | 0.012 | 98.2% | 172 s |

To rebuild the set, call `gen(20000)` from `bench.py` right after `torch.manual_seed(0)`, then split at index 18,000. Generation takes about 7.5 s on 2 CPU cores. The script lives in the sandbox only; ask if you want it saved to your folder.
