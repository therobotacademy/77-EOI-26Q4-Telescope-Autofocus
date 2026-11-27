# Next step (as of 2026-11-27)

## Done: report 06, signed defocus and robustness (week 4)

Gate passed in simulation: sign accuracy 98.2% for |Δz| ≥ 200 µm. See `REPORTS/06-signed-defocus-robustness.md`.

## Next

- Agree the near-focus limit and the sign target.
- **Report 07, real images (week 5):** MILAN frames for blur size. Testing the sign needs a real focuser sweep.
- **Report 08 (week 6):** simulated closed-loop autofocus and final conclusions. This can start now, without real data.
- Optional: test how sensitive the sign is to the assumed spherical aberration.

## Decisions still open

- Real telescope and camera specs. Changing them may mean regenerating the data and repeating reports 03–06.
- Unit for Δz, near-focus limit, depth-of-focus definition and final targets.
- Whether a focuser sweep can be captured. Ask early.

## Repository

- Branch `feat/hfr-baseline-reports`: merged into `master` through pull request 1 (merge commit `fbe3049`). The branch is kept.
- Committed in `8e92d92`: the report 05 annexes, report 06, `signed_robustness.py`, `weights/` and the README updates.
- `antigravity/` and `docs/01-hfr-baseline.md` are committed on branch `docs/notes-and-next`.
