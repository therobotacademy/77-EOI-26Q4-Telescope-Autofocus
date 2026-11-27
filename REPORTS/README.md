# Progress reports

Sequential reports that track the project. Files are named `NN-topic.md` and are numbered in the order
the work is reported. A number is never reused.

## Every report contains

1. **Objective and gate**: what was to be shown, and the pass criterion.
2. **Method**: data, scripts and commands needed to reproduce it.
3. **Results**: tables and figures, only from runs that were actually executed.
4. **Discussion**: what the results mean, limits and open doubts.
5. **Next steps**: actions that follow, and the report they feed.

## List of reports

| # | File | Content | Plan week | Work done | Report written |
|---|---|---|---|---|---|
| 01 | [`01-problem-and-data-survey.md`](01-problem-and-data-survey.md) | Problem definition, data sources, labelling, success metrics | before 1 | Yes | Yes |
| 02 | [`02-sandbox-and-toy-benchmark.md`](02-sandbox-and-toy-benchmark.md) | CPU timing and toy set v0 (not usable as a result) | before 1 | Yes | Yes |
| 03 | [`03-physical-simulator.md`](03-physical-simulator.md) | Optical model, calibration on one real frame, dataset v1, is the sign visible | 1 | Prototype, assumed optics | Yes |
| 04 | [`04-hfr-baseline.md`](04-hfr-baseline.md) | HFR error per \|Δz\| bin on the synthetic test set | 2 | Yes | Yes |
| 05 | [`05-cnn-blur-size.md`](05-cnn-blur-size.md) | CNN against HFR on \|Δz\| | 3 | Yes | Yes |
| 06 | `06-signed-defocus-robustness.md` | Sign accuracy and error across seeing, jitter and SNR | 4 | Initial check only | No |
| 07 | `07-real-images.md` | MILAN frames and a focuser sweep, real-to-synthetic gap | 5 | No | No |
| 08 | `08-autofocus-loop-and-final.md` | Closed-loop moves to focus, overall conclusions | 6 | No | No |

Unplanned work (for example, recalibration with the real telescope's optics) takes the next free number
and is added to this table.
