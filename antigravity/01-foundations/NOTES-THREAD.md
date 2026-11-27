# 1 - `<TITLE IN CAPITAL LETTERS></title>`

- Create when a new topic is detected and add +1 when you detect a new topic (similar to what is a book chapter)
- You can ad subparagraphs ## 1.1 like sections in a book chapter. Do it only when it's clearly a subtopic
- Format as shown below

> Prompt

Literal answer HERE

Separator is the standard ---

---

Do not remove the template instructions above. Start HERE

# 1 - TRAINED CNN STATUS AND ARCHITECTURE

> What CNN have you already trained?

No trained CNN weights or checkpoints are saved in this repository (there are no `.pt`/`.pth` files). So far, only **one compact 4-block CNN (~249k parameters)** has been trained as a short sanity check—run twice during the earlier sandbox session documented in [README.md](file:///c:/Users/brjap/Documents/__CODE_gpu/___PROJECT_8-EOI/77(public)-EOI-26Q4-Telescope-Autofocus/README.md) and [WORKFLOW_PROCEDURE.md](file:///c:/Users/brjap/Documents/__CODE_gpu/___PROJECT_8-EOI/77(public)-EOI-26Q4-Telescope-Autofocus/WORKFLOW_PROCEDURE.md).

### Architecture ([train_check_v1.py](file:///c:/Users/brjap/Documents/__CODE_gpu/___PROJECT_8-EOI/77(public)-EOI-26Q4-Telescope-Autofocus/Fourier-optics%20simulator/train_check_v1.py#L22-L28))
A sequential PyTorch CNN taking a single `1×64×64` star cut-out and regressing a single scalar signed focus error:

- **Block 1:** `Conv2d(1 → 32, 3×3, pad=1)` → `ReLU` → `MaxPool2d(2)`
- **Block 2:** `Conv2d(32 → 64, 3×3, pad=1)` → `ReLU` → `MaxPool2d(2)`
- **Block 3:** `Conv2d(64 → 128, 3×3, pad=1)` → `ReLU` → `MaxPool2d(2)`
- **Block 4:** `Conv2d(128 → 128, 3×3, pad=1)` → `ReLU` → `AdaptiveAvgPool2d(1)`
- **Head:** `Flatten` → `Linear(128 → 64)` → `ReLU` → `Linear(64 → 1)` (**248,577 parameters**)

### Runs Executed So Far

| Run | Script | Dataset | Training Setup | Reported Result | Status / Caveat |
|---|---|---|---|---|---|
| **Physical Dataset v1 check** | [train_check_v1.py](file:///c:/Users/brjap/Documents/__CODE_gpu/___PROJECT_8-EOI/77(public)-EOI-26Q4-Telescope-Autofocus/Fourier-optics%20simulator/train_check_v1.py) | [defocus_stars_v1.npz](file:///c:/Users/brjap/Documents/__CODE_gpu/___PROJECT_8-EOI/77(public)-EOI-26Q4-Telescope-Autofocus/Fourier-optics%20simulator/defocus_stars_v1.npz) (16k train / 2k test) | L1 loss, Adam `1e-3`, batch 128, **4 epochs** (183 s on 2 CPUs), 4-px border median/std normalization | **148 µm** test MAE; **88.9%** overall sign acc (`58%` for `0–200 µm`, `90%` for `200–400 µm`, `95%` for `400–800 µm`, `98%` for `> 800 µm`) | Sanity check only (does not save weights, evaluates on `test_idx` instead of `val_idx`, uses placeholder 200 mm f/5 optics + 0.4 waves spherical aberration) |
| **Toy Benchmark v0 timing check** | `bench.py` *(sandbox-only, not in repo)* | [toy_defocus_stars_v0.npz](file:///c:/Users/brjap/Documents/__CODE_gpu/___PROJECT_8-EOI/77(public)-EOI-26Q4-Telescope-Autofocus/Benchmark%20v0/toy_defocus_stars_v0.npz) (18k train / 2k test) | MSE loss, Adam `1e-3`, batch 128, **3 epochs** (172 s on 2 CPUs) | **0.012** test MAE (normalized `[-1, 1]`), **98.2%** sign acc ([Dataset Card](file:///c:/Users/brjap/Documents/__CODE_gpu/___PROJECT_8-EOI/77(public)-EOI-26Q4-Telescope-Autofocus/Dataset%20Card%20—%20Toy%20Defocused-Star%20Set%20%28benchmark%20v0%29.md#L61-L70)) | Non-physical shortcut (`w = 1.5 + 0.5·d`); for CPU throughput/loop smoke-testing only |

No production or Week 3–4 model (e.g., tuned against `val_idx`, compared against an HFR baseline, or saved to disk) has been trained yet.

---
