# 01 · Problem definition and data survey

Work dated 2026-11-18. Written 2026-11-27 from the [problem statement](<../Single-Image%20Telescope%20Autofocus%20—%20Problem%20Statement%20&%20Plan.md>) and the [workflow log](../WORKFLOW_PROCEDURE.md).

## Objective and gate

Define the task and find data to solve it. No plan gate: this precedes week 1.

## Method

- 12 web searches (Kaggle and open web) for defocus, PSF, star-field, focus-stack and FITS data.
- Labelling routes compared: simulator, real focuser sweep, unlabelled real frames.

## RESULTS

**Task:** from one 64×64 star cut-out, estimate the signed focus error Δz = p − p* (focuser position minus best focus).

| Source | Possible role | Focus label |
|---|---|---|
| Own simulator | Train, validation, test | Exact signed Δz |
| Own focuser sweep (FITS) | Real labelled test | FOCPOS − V-curve best focus |
| MILAN Sky Survey | Real unlabelled test | Blur size only (HFR) |
| BBBC006 (microscopy) | Optional parallel benchmark | Signed z, 2 µm steps |
| Kaggle: Mapping Dark Matter, Blur Dataset | Reference only | None |

No dataset found with stars labelled by signed focus error.

**Success metrics:**

| Metric | Target |
|---|---|
| MAE of Δz per \|Δz\| bin, synthetic test | Beat the HFR baseline |
| Sign accuracy outside the near-focus zone | Proposed ≥ 95% |
| Sign accuracy on a real sweep | Reported, no target |
| Blur size against HFR on MILAN | Reported as correlation |
| Moves to reach focus, simulated | Proposed 1–2 |

## Discussion

- Training has to be synthetic, so every conclusion depends on how faithful the simulator is.
- The sign is the hard part. Symmetric defocus looks the same on both sides of focus; an asymmetry such as spherical aberration is required.
- The searches could not query Kaggle's own catalogue directly, so "no dataset" means none was found.
- Targets are proposals. The unit of Δz, the near-focus zone and the depth of focus are not formally fixed.

## Next steps

- Agree the telescope to model, the unit of Δz and the numerical targets.
- Ask for a real focuser sweep.
- Check that the sandbox can train a model (report [02](02-sandbox-and-toy-benchmark.md)) and build the simulator (report [03](03-physical-simulator.md)).
