# Toy defocused-star set (benchmark v0)

**Benchmark placeholder only. Do not use it to train or evaluate the autofocus model.**
The sign of the label is encoded by an invented rule (ring width = 1.5 + 0.5·d), not by real optics.

## Contents of `toy_defocus_stars_v0.npz`

| Key | Shape, type | Meaning |
|---|---|---|
| `counts` | (20000, 1, 64, 64), uint16 | Photon counts per pixel (0 to 280) |
| `scale` | scalar, float32 | 200. Image as in the card = `counts / scale` (0.0 to 1.40) |
| `d` | (20000,), float32 | Signed, normalised defocus in [-1, 1]; 50.3% positive |
| `train_idx` | (18000,), int | Indices 0 to 17,999 |
| `test_idx` | (2000,), int | Indices 18,000 to 19,999 |

## Load

```python
import numpy as np
z = np.load("toy_defocus_stars_v0.npz")
X = z["counts"].astype(np.float32) / z["scale"]
d = z["d"]
X_train, d_train = X[z["train_idx"]], d[z["train_idx"]]
X_test,  d_test  = X[z["test_idx"]],  d[z["test_idx"]]
```

## Generation (seed 0)

Ring radius R = 2 + 12·|d| px, width w = 1.5 + 0.5·d px, centred at (32, 32);
counts = Poisson(200·exp(-(r-R)²/(2w²)) + 10). Rebuild with `python make_toy_defocus_stars.py`
(needs numpy and torch; torch's random generator is what makes it reproducible).

No optics, seeing, read noise, saturation or off-centre framing. Full dataset card:
https://claude.ai/artifact/BeuQhoPjsrbDBQujVyrooj
