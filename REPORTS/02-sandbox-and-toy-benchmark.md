# 02 · Sandbox timing and toy benchmark

Work dated 2026-11-18. Written 2026-11-27 from the [dataset card](<../Dataset%20Card%20—%20Toy%20Defocused-Star%20Set%20%28benchmark%20v0%29.md>). Figures are as logged in that session and were not re-run.

## Objective and gate

Find the image and network size that a CPU-only sandbox can train, and check the training loop. No plan gate.

## Method

- Hardware: 2 Xeon cores (2.1 GHz), 7.8 GB RAM, no GPU, PyTorch 2.15 (CPU).
- Toy set: 20,000 Gaussian rings, radius 2 + 12·|d| px, width 1.5 + 0.5·d px, Poisson noise ([Benchmark v0](../Benchmark%20v0/README.md)).
- 4-block CNN, MSE loss, Adam 1e-3, batch 128, 3 epochs. The script `bench.py` is not in the repository.

## RESULTS

| Image size | Training speed | One epoch of 20k |
|---|---|---|
| 64×64 | about 340 images/s | about 1 min |
| 128×128 | about 60 images/s | about 5.5 min |

| Epoch | Test MAE (d, range −1 to 1) | Sign accuracy | Elapsed |
|---|---|---|---|
| 1 | 0.021 | 97.8% | 57 s |
| 2 | 0.013 | 98.3% | 114 s |
| 3 | 0.012 | 98.2% | 172 s |

## Discussion

- The training loop works, and 64×64 with a small CNN fits the sandbox.
- **The accuracy figures say nothing about autofocus.** The sign is encoded by an invented ring-width rule, with no optics, seeing or sensor model.
- The speed limits are those of the sandbox. A local RTX 2060 is now available (conda env `xgnn`), so they no longer bound training length.

## Next steps

- Keep the toy set for smoke tests only.
- Replace it with a physical simulator (report [03](03-physical-simulator.md)).
