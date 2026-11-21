# Single-Image Telescope Autofocus — Step-by-Step Generation Procedure & Prompts

> **Generated Date**: 2026-11-21
> **Session distilled**: one Claude session, 2026-11-18, 10:14–11:22 (Europe/Madrid), Project "EOI Deep Learning"
> **Methodology**: synthesis of the live conversation (verbatim prompts), sandbox file inventory and tool outputs.
> **Sources available**: the conversation itself and the files in this folder. There is no `transcript.jsonl` and no git repository, so commit columns read "none" throughout.

---

## 1. Executive Summary & Process Milestones

| Phase | Focus Area | Key Actions Executed | Primary Artifacts |
|---|---|---|---|
| **Phase 1** | Data source survey | Searched Kaggle and the open web for star images labelled with focus error; none exists on Kaggle | Ranked list (in chat) |
| **Phase 2** | Sandbox capability & labelling requirement | Measured hardware, installed CPU PyTorch, timed a CNN; defined how images get focus labels | `bench.py` |
| **Phase 3** | Problem statement & plan | Wrote the problem, data/labels, constraints, metrics, 6-week gated plan, risks | Docs: *Problem Statement & Plan* |
| **Phase 4** | Toy dataset documentation & release | Dataset card, downloadable `.npz`, sample grid and plotting script for the benchmark set | Docs: *Dataset Card v0*, [`toy_defocus_stars_v0.npz`](toy_defocus_stars_v0.npz) |
| **Phase 5** | Plan feasibility check | Mapped each plan week to what runs in the sandbox; confirmed Zenodo/BBBC access; installed astropy, poppy | Feasibility table (in chat) |
| **Phase 6** | Physical simulator calibrated on a real frame | Fourier-optics star simulator, fit to the real frame's radial profile, dataset v1 (20k), training sanity check | [`star_sim.py`](star_sim.py), [`defocus_stars_v1.npz`](defocus_stars_v1.npz) |
| **Phase 7** | Process distillation | This document | [`WORKFLOW_PROCEDURE.md`](WORKFLOW_PROCEDURE.md) |

---

## 2. Process Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor Tutor as Bernardo - tutor
    participant Agent as Claude
    participant Web as Web - Kaggle Zenodo BBBC
    participant Box as Sandbox - 2 CPU no GPU
    participant Docs as Claude Docs artifacts

    Tutor->>Agent: P-01 Find Kaggle datasets for our purpose
    Agent->>Web: Search star defocus PSF and focus datasets
    Web-->>Agent: No signed focus labels on Kaggle
    Agent-->>Tutor: Ranked list - synthetic train, real test

    Tutor->>Agent: P-02 Hardware available and does it fit
    Agent->>Box: Inspect CPU RAM disk, install PyTorch, time CNN
    Tutor->>Agent: P-03 Images must be tagged with focus
    Agent-->>Tutor: Fits at 64x64, labels from simulator or focuser sweep

    Tutor->>Agent: P-04 Problem statement and plan
    Agent->>Docs: Problem Statement and Plan with 6-week gated roadmap

    Tutor->>Agent: P-05 to P-09 Card, download, images, feasibility, script
    Agent->>Docs: Dataset Card for toy set v0
    Agent->>Box: Save npz, plot samples, check Zenodo and BBBC access
    Agent-->>Tutor: Files, grid image, feasibility per week

    Tutor->>Agent: P-10 Real frame - make synthetic match it
    Agent-->>Tutor: Feature by feature fix proposal
    Tutor->>Agent: P-11 Go ahead
    Agent->>Box: Build Fourier-optics simulator
    Agent->>Box: Fit defocus and seeing to real radial profile
    Agent->>Box: Generate 20k images in shards, regenerate after review
    Agent->>Box: Train CNN check - 89 percent sign accuracy
    Agent-->>Tutor: Dataset v1, README, scripts, comparison images

    Tutor->>Agent: P-12 Distill the history
    Agent-->>Tutor: WORKFLOW_PROCEDURE.md
```

---

## 3. Detailed Chronological Phases & Exact Prompts

### Phase 1: Data source survey

* **Objective**: Find existing datasets usable for single-image, signed-defocus estimation of star images.
* **Prompt Used** (P-01):
  ```text
  Look for Kaggle datasets that serves for our purposes. Build a quick list for choosing on
  ```
* **Actions Executed**:
  1. Read the project memory: the task is to estimate a star's signed defocus with a CNN trained on synthetic data.
  2. Ran 12 web searches (Kaggle-restricted and open web) for defocus, PSF, star field, focus-stack and FITS datasets.
  3. Ranked candidates by fit. On Kaggle: Mapping Dark Matter (48×48 PSF-convolved star/galaxy pairs, medium fit), Blur Dataset (350 sharp/defocused/motion triplets, low), SpaceNet and NASA APOD (low). Off Kaggle: BBBC006 (microscopy z-stack with signed z, high as a parallel task), MILAN Sky Survey (raw Stellina FITS, high as a real test set), Integer L0 (multi-observatory raw FITS).
* **Artifacts & Files Produced**: ranked list in chat (no file).
* **Commands & Verification**: `WebSearch` × 12; Kaggle's own catalogue could not be queried directly.
* **Outcome**: no Kaggle dataset carries signed focus labels, so training stays synthetic and real data is for testing.

### Phase 2: Sandbox capability & labelling requirement

* **Objective**: Decide whether the sandbox can run the experiments, and fix how images get focus labels.
* **Prompt Used** (P-02, then P-03 sent mid-turn):
  ```text
  I want to make the tests here in the sandbox. What is the hardware available, and does it fit for our needs?
  ```
  ```text
  Since the problem is to solve focus error of the telescope, we need to tag images with the focus feature
  ```
* **Actions Executed**:
  1. Measured the hardware: 2 cores (Xeon 2.1 GHz, AVX-512, AMX), 7.8 GB RAM, ~31 GB free disk, no GPU.
  2. Installed PyTorch 2.15 (CPU) and wrote `bench.py` (toy ring generator + 4-block CNN).
  3. Timed training: 64×64 ≈ 340 images/s (≈1 min per 20k-image epoch); 128×128 ≈ 60 images/s (≈5.5 min).
  4. Answered P-03: labels come for free from a simulator; real labels need a focuser sweep (FITS `FOCPOS` minus the V-curve best focus); unlabelled real sets give blur size only.
* **Artifacts & Files Produced**: [`../bench.py`](../bench.py)
* **Commands & Verification**:
  ```bash
  nproc; free -h; df -h; nvidia-smi; lscpu
  pip install --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu
  python3 bench.py
  ```
* **Outcome**: the sandbox fits if cut-outs stay at 64×64 (≤96×96), the CNN stays small and runs finish within about 10 min.

### Phase 3: Problem statement & plan

* **Objective**: Write down the problem as defined so far and a plan to solve it.
* **Prompt Used** (P-04):
  ```text
  Write the statement of the problem defined so far, and write a plan for solving
  ```
* **Actions Executed**:
  1. Created a Claude Docs document with six sections: problem statement (Δz = p − p*), data and labels, constraints, success criteria, plan, risks.
  2. Drew the plan as a 6-week roadmap with one gate per week (simulator → dataset and HFR baseline → CNN for blur size → sign and robustness → real images → autofocus loop).
  3. Flagged the critical risk: with symmetric optics the sign is not observable, so the choice of telescope to model is the first decision.
* **Artifacts & Files Produced**: Docs: [Single-Image Telescope Autofocus — Problem Statement & Plan](https://claude.ai/artifact/VZCyanFEqg8PTc9Ab85Lb2)
* **Commands & Verification**: rendered the roadmap diagram and checked it visually.
* **Outcome**: an agreed reference plan with four open decisions (telescope, Δz unit, targets, focuser sweep).

### Phase 4: Toy dataset documentation & release

* **Objective**: Document, release and visualise the benchmark data made in Phase 2.
* **Prompt Used** (P-05, P-06, P-07, P-09):
  ```text
  Write the dataset card of the synthetic data you produced
  ```
  ```text
  Give the dataset for downloading
  ```
  ```text
  Show me several images from the dataset
  ```
  ```text
  Give me the script you used for visualization
  ```
* **Actions Executed**:
  1. Regenerated the seed-0 toy set and measured its statistics (20,000 × 64×64; values 0–1.40; labels ±1, 50.3% positive; split 18,000 / 2,000).
  2. Wrote the dataset card, stating plainly that the sign is encoded by an invented width rule, so the toy set is only good for timing and testing code.
  3. Saved the set as exact uint16 photon counts plus `scale = 200` (60 MB), with a README and a rebuild script; reloaded and checked it against the card.
  4. Plotted 12 stars from d = −1 to +1, fixed overlapping labels, and released the plotting script.
* **Artifacts & Files Produced**:
  - Docs: [Dataset Card — Toy Defocused-Star Set (benchmark v0)](https://claude.ai/artifact/BeuQhoPjsrbDBQujVyrooj)
  - [`toy_defocus_stars_v0.npz`](toy_defocus_stars_v0.npz), [`README.md`](README.md), [`make_toy_defocus_stars.py`](make_toy_defocus_stars.py)
  - [`toy_samples_v0.png`](toy_samples_v0.png), [`plot_toy_samples.py`](plot_toy_samples.py)
* **Commands & Verification**:
  ```bash
  python3 make_toy_defocus_stars.py          # writes toy_defocus_stars_v0.npz
  python3 plot_toy_samples.py toy_defocus_stars_v0.npz toy_samples_v0.png
  ```
* **Outcome**: benchmark set v0 released with its limits stated (3-epoch baseline: MAE 0.012, 98.2% sign, which only shows the loop trains).

### Phase 5: Plan feasibility check

* **Objective**: Say which plan weeks can be executed in the sandbox now.
* **Prompt Used** (P-08, sent mid-turn; typo kept):
  ```text
  Which part of the plan can I execute now in the sandblox?
  ```
* **Actions Executed**:
  1. Checked network access: Zenodo (MILAN) and the Broad Institute site (BBBC006) are reachable.
  2. Installed astropy (FITS) and poppy (optics).
  3. Mapped the weeks: 1–4 fully, 5 partly (the real focuser sweep needs a telescope), 6 in simulation only.
* **Artifacts & Files Produced**: feasibility table (in chat).
* **Commands & Verification**:
  ```bash
  curl -s -o /dev/null -w "%{http_code}" https://zenodo.org/records/7304607
  pip install --break-system-packages astropy poppy
  ```
* **Outcome**: only one decision blocks week 1: which telescope to model.

### Phase 6: Physical simulator calibrated on a real frame

* **Objective**: Replace the toy generator with physical star images that match a real raw frame.
* **Prompt Used** (P-10 with a screenshot of a raw "ADU" frame and its "preprocesada" version, then P-11):
  ```text
  Look at what actual images should look. Left is the raw image. Propose how to fix the dataset so that synthetic images matches this pattern
  ```
  ```text
  Go ahead
  ```
* **Actions Executed**:
  1. Compared the real frame with the toy set feature by feature (wide donut, partly filled hole, mottled ring, low SNR, off-centre star, sign cue) and proposed a fix for each.
  2. Built `star_sim.py`: pupil with 30% obstruction, defocus + 0.4 waves spherical aberration + random astigmatism/coma + Kolmogorov turbulence, |FFT|² PSF, camera binning, Moffat seeing, ADU sensor model; typical 200 mm f/5 optics with 3.76 µm pixels as placeholders.
  3. Converted the screenshot's left panel to a 64×64 array and measured it: ring peak at r = 14 px, half-max from 5 to 23 px, hole at 42% of peak, ring SNR ≈ 6.5. Fitted the noise-free simulator to its radial profile: best at 9 waves, seeing FWHM 10 px, β 4, D/r0 6 (RMS error 0.04).
  4. Generated 20,000 images in shards of 2,000 after a long foreground run was killed. Reviewed the sample grid, then narrowed the ranges (±11 waves = ±1,210 µm so the donut fits; seeing 6–12 px and D/r0 2–5 to remove sharp speckle) and regenerated (398 s, 109 MB).
  5. Training check, 4 epochs (183 s): MAE 148 µm, sign accuracy 88.9% overall; 58% under 200 µm, 90% at 200–400, 95% at 400–800, 98% above 800 µm.
* **Artifacts & Files Produced**:
  - [`star_sim.py`](star_sim.py), [`make_dataset_v1.py`](make_dataset_v1.py), [`plot_samples_v1.py`](plot_samples_v1.py), [`train_check_v1.py`](train_check_v1.py), [`../fit_profile.py`](../fit_profile.py)
  - [`defocus_stars_v1.npz`](defocus_stars_v1.npz), [`README_v1.md`](README_v1.md), [`real_frame_grey64.npy`](real_frame_grey64.npy)
  - [`sim_vs_real_v1.png`](sim_vs_real_v1.png), [`samples_v1.png`](samples_v1.png), logs [`gen_v1.log`](gen_v1.log), [`train_v1.log`](train_v1.log)
* **Commands & Verification**:
  ```bash
  PYTHONPATH=. python3 ../fit_profile.py                        # profile fit, ~18 s per grid
  PYTHONPATH=. nohup python3 make_dataset_v1.py 20000 > gen_v1.log &
  python3 plot_samples_v1.py                                     # visual review -> narrowed ranges
  PYTHONPATH=. nohup python3 train_check_v1.py defocus_stars_v1.npz 4 > train_v1.log &
  ```
* **Outcome**: synthetic stars match the real frame's profile and noise; the sign is learnable away from focus. Open items: real optics, the true spherical aberration (needs a focuser sweep), raw FITS for calibration, the real preprocessing.

### Phase 7: Process distillation

* **Objective**: Record the session as a reproducible procedure.
* **Prompt Used** (P-12):
  ```text
  /anthropic-skills:history-workflow-distiller
  ```
* **Actions Executed**: inventoried the sandbox files (all present after the pause), extracted the 12 prompts verbatim from the conversation, wrote this document.
* **Artifacts & Files Produced**: [`WORKFLOW_PROCEDURE.md`](WORKFLOW_PROCEDURE.md)
* **Outcome**: complete procedure with Annex A.

---

## 4. Git Revision History & Traceability

No git repository exists in the sandbox, so there are no commits. Files carry their modification times instead:

| File | Last modified (2026-11-18, Madrid) | Phase |
|---|---|---|
| `../bench.py` | 10:19 | 2 |
| `toy_defocus_stars_v0.npz`, `make_toy_defocus_stars.py`, `README.md` | 10:42 | 4 |
| `plot_toy_samples.py`, `toy_samples_v0.png` | 10:45–10:46 | 4 |
| `real_frame_grey64.npy`, `../fit_profile.py`, `sim_vs_real_v1.png` | 10:53–10:55 | 6 |
| `make_dataset_v1.py`, `train_check_v1.py` | 11:02–11:03 | 6 |
| `star_sim.py`, `README_v1.md`, `plot_samples_v1.py` | 11:11 | 6 |
| `defocus_stars_v1.npz`, `samples_v1.png`, `train_v1.log` | 11:17–11:21 | 6 |

To version this work, run `git init` in this folder and commit using your own git identity.

---

## 5. Verification & Reproducibility Summary

- [x] All 7 phases documented with objectives and verbatim prompts.
- [x] Mermaid sequence diagram covers all 12 prompts in order.
- [x] File references are relative to this folder (`../` for the two scripts in its parent); Docs artifacts linked by URL.
- [x] Every figure quoted was read from a tool output or a logged file in the session.
- [ ] Git traceability: not available (no repository).
- [ ] Wall-clock times per prompt are message times; intermediate tool timings come from logs where they exist.

---

## Annex A: Master Prompts Catalog & Traceability Matrix

> All 12 user prompts of the session, verbatim. Times are Europe/Madrid, 2026-11-18 unless noted. Commits: none (no repository).

### A.1 Chronological Prompt Index

| # | Prompt ID | Time | User Prompt Summary | Primary Target / Phase |
|---|---|---|---|---|
| 01 | **P-01** | 10:14 | Find Kaggle datasets and build a quick list | Data survey · Phase 1 |
| 02 | **P-02** | 10:18 | Sandbox hardware and whether it fits | `bench.py` · Phase 2 |
| 03 | **P-03** | 10:21 (mid-turn) | Images must be tagged with the focus value | Labelling · Phase 2 |
| 04 | **P-04** | 10:27 | Problem statement and plan | Docs plan · Phase 3 |
| 05 | **P-05** | 10:37 | Dataset card of the synthetic data | Docs card · Phase 4 |
| 06 | **P-06** | 10:41 | Dataset for downloading | `toy_defocus_stars_v0.npz` · Phase 4 |
| 07 | **P-07** | 10:45 | Show several dataset images | `toy_samples_v0.png` · Phase 4 |
| 08 | **P-08** | 10:45 (mid-turn) | Which plan parts run in the sandbox | Feasibility · Phase 5 |
| 09 | **P-09** | 10:46 | Visualisation script | `plot_toy_samples.py` · Phase 4 |
| 10 | **P-10** | 10:50 | Make synthetic images match a real raw frame (screenshot) | Fix proposal · Phase 6 |
| 11 | **P-11** | 10:51 | Go ahead | `star_sim.py`, dataset v1 · Phase 6 |
| 12 | **P-12** | 2026-11-21 12:34 | Distill the history into a procedure | This document · Phase 7 |

---

### A.2 Exhaustive Verbatim Prompt Catalog

#### Prompt P-01
* **Metadata**: 10:14 | Target: dataset survey
* **Prompt Text**:
  ```text
  Look for Kaggle datasets that serves for our purposes. Build a quick list for choosing on
  ```
* **User Intent & Objective**: find ready-made data for the telescope autofocus project and choose among options.
* **Technical Actions & Artifacts Produced**:
  - 12 web searches; ranked table of 4 Kaggle and 3 non-Kaggle datasets; recommendation: synthetic for training, MILAN and BBBC006 for testing.
  - No files.
* **Associated Commits**: none

#### Prompt P-02
* **Metadata**: 10:18 | Target: sandbox
* **Prompt Text**:
  ```text
  I want to make the tests here in the sandbox. What is the hardware available, and does it fit for our needs?
  ```
* **User Intent & Objective**: know whether experiments can run in the sandbox.
* **Technical Actions & Artifacts Produced**:
  - Hardware inspection, CPU PyTorch install, CNN throughput benchmark.
  - [`../bench.py`](../bench.py)
* **Associated Commits**: none

#### Prompt P-03
* **Metadata**: 10:21, sent while P-02 was running | Target: labelling strategy
* **Prompt Text**:
  ```text
  Since the problem is to solve focus error of the telescope, we need to tag images with the focus feature
  ```
* **User Intent & Objective**: make labelled focus error a hard requirement of any dataset.
* **Technical Actions & Artifacts Produced**:
  - Defined three label routes: synthetic (exact signed Δz), real focuser sweep (`FOCPOS` minus V-curve best focus), unlabelled real (size only).
  - Answered in chat; no files.
* **Associated Commits**: none

#### Prompt P-04
* **Metadata**: 10:27 | Target: Claude Docs
* **Prompt Text**:
  ```text
  Write the statement of the problem defined so far, and write a plan for solving
  ```
* **User Intent & Objective**: a written problem definition and plan to work from.
* **Technical Actions & Artifacts Produced**:
  - Six-section doc with a 6-week gated roadmap diagram.
  - [Problem Statement & Plan](https://claude.ai/artifact/VZCyanFEqg8PTc9Ab85Lb2)
* **Associated Commits**: none

#### Prompt P-05
* **Metadata**: 10:37 (viewing the Problem Statement doc) | Target: Claude Docs
* **Prompt Text**:
  ```text
  Write the dataset card of the synthetic data you produced
  ```
* **User Intent & Objective**: document the synthetic data made so far.
* **Technical Actions & Artifacts Produced**:
  - Regenerated the toy set to measure its statistics; five-section card (summary, generation, structure, intended use and limitations, baseline and reproduction).
  - [Dataset Card — Toy Defocused-Star Set (benchmark v0)](https://claude.ai/artifact/BeuQhoPjsrbDBQujVyrooj)
* **Associated Commits**: none

#### Prompt P-06
* **Metadata**: 10:41 | Target: dataset file
* **Prompt Text**:
  ```text
  Give the dataset for downloading
  ```
* **User Intent & Objective**: get the data as a file.
* **Technical Actions & Artifacts Produced**:
  - Saved as uint16 counts + scale (exact, 60 MB), reloaded and checked against the card.
  - [`toy_defocus_stars_v0.npz`](toy_defocus_stars_v0.npz), [`README.md`](README.md), [`make_toy_defocus_stars.py`](make_toy_defocus_stars.py)
* **Associated Commits**: none

#### Prompt P-07
* **Metadata**: 10:45 | Target: visualisation
* **Prompt Text**:
  ```text
  Show me several images from the dataset
  ```
* **User Intent & Objective**: see what the data looks like.
* **Technical Actions & Artifacts Produced**:
  - 2×6 grid from d = −1 to +1, re-rendered once to fix overlapping labels.
  - [`toy_samples_v0.png`](toy_samples_v0.png)
* **Associated Commits**: none

#### Prompt P-08
* **Metadata**: 10:45, sent while P-07 was running | Target: plan feasibility
* **Prompt Text**:
  ```text
  Which part of the plan can I execute now in the sandblox?
  ```
* **User Intent & Objective**: know what can start immediately.
* **Technical Actions & Artifacts Produced**:
  - Network checks (Zenodo, BBBC), astropy and poppy installed; per-week feasibility table in chat.
* **Associated Commits**: none

#### Prompt P-09
* **Metadata**: 10:46 | Target: plotting script
* **Prompt Text**:
  ```text
  Give me the script you used for visualization
  ```
* **User Intent & Objective**: reuse the plotting code.
* **Technical Actions & Artifacts Produced**:
  - Cleaned, commented, command-line script; re-run to confirm the same 12 images.
  - [`plot_toy_samples.py`](plot_toy_samples.py)
* **Associated Commits**: none

#### Prompt P-10
* **Metadata**: 10:50 | Attachment: screenshot with a raw "ADU" frame and its "preprocesada" version | Target: simulator design
* **Prompt Text**:
  ```text
  Look at what actual images should look. Left is the raw image. Propose how to fix the dataset so that synthetic images matches this pattern
  ```
* **User Intent & Objective**: make synthetic data look like real telescope frames.
* **Technical Actions & Artifacts Produced**:
  - Feature-by-feature comparison table, 6-step physical generation recipe, calibration method and data requests (raw FITS, optics, preprocessing). Answered in chat.
* **Associated Commits**: none

#### Prompt P-11
* **Metadata**: 10:51 | Target: simulator and dataset v1
* **Prompt Text**:
  ```text
  Go ahead
  ```
* **User Intent & Objective**: build the proposed simulator with typical values now.
* **Technical Actions & Artifacts Produced**:
  - Simulator, profile fit to the real frame, 20k dataset in shards, review and regeneration, training check (88.9% sign accuracy).
  - [`star_sim.py`](star_sim.py), [`make_dataset_v1.py`](make_dataset_v1.py), [`defocus_stars_v1.npz`](defocus_stars_v1.npz), [`README_v1.md`](README_v1.md), [`plot_samples_v1.py`](plot_samples_v1.py), [`train_check_v1.py`](train_check_v1.py), [`../fit_profile.py`](../fit_profile.py), [`real_frame_grey64.npy`](real_frame_grey64.npy), [`sim_vs_real_v1.png`](sim_vs_real_v1.png), [`samples_v1.png`](samples_v1.png)
* **Associated Commits**: none

#### Prompt P-12
* **Metadata**: 2026-11-21 12:34 | Target: procedure document
* **Prompt Text**:
  ```text
  /anthropic-skills:history-workflow-distiller
  ```
* **User Intent & Objective**: a reproducible record of how the work was produced.
* **Technical Actions & Artifacts Produced**:
  - File inventory, prompt extraction, this document: [`WORKFLOW_PROCEDURE.md`](WORKFLOW_PROCEDURE.md)
* **Associated Commits**: none

---

### A.3 Comprehensive Traceability Matrix (Prompt → Phase → Artifacts → Commits)

| Prompt ID | Phase | Target Module / Repo | Produced Artifacts | Git Commit(s) |
|---|---|---|---|---|
| **P-01** | Phase 1 | Data survey | Ranked list (chat) | none |
| **P-02** | Phase 2 | Sandbox benchmark | `../bench.py` | none |
| **P-03** | Phase 2 | Labelling strategy | Label routes (chat) | none |
| **P-04** | Phase 3 | Claude Docs | Problem Statement & Plan doc | none |
| **P-05** | Phase 4 | Claude Docs | Dataset Card v0 doc | none |
| **P-06** | Phase 4 | Toy dataset | `toy_defocus_stars_v0.npz`, `README.md`, `make_toy_defocus_stars.py` | none |
| **P-07** | Phase 4 | Visualisation | `toy_samples_v0.png` | none |
| **P-08** | Phase 5 | Feasibility | Per-week table (chat) | none |
| **P-09** | Phase 4 | Visualisation | `plot_toy_samples.py` | none |
| **P-10** | Phase 6 | Simulator design | Fix proposal (chat) | none |
| **P-11** | Phase 6 | Simulator + dataset v1 | `star_sim.py`, `make_dataset_v1.py`, `defocus_stars_v1.npz`, `README_v1.md`, `plot_samples_v1.py`, `train_check_v1.py`, `../fit_profile.py`, `real_frame_grey64.npy`, `sim_vs_real_v1.png`, `samples_v1.png` | none |
| **P-12** | Phase 7 | Documentation | `WORKFLOW_PROCEDURE.md` | none |
