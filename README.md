# GARP-PAD: A Gated Attention Rotation-equivariant Patch-based Architecture for Fingerprint Presentation Attack Detection

Official PyTorch implementation of **GARP-PAD**, a compact, rotation-equivariant,
patch-based architecture for **Fingerprint Presentation Attack Detection (PAD)**.

**Simone Carta, Roberto Casula, Gian Luca Marcialis**
Department of Electrical and Electronic Engineering, University of Cagliari, Italy
`{simone.carta97, roberto.casula, marcialis}@unica.it`

> 📄 To appear at the **IEEE International Joint Conference on Biometrics (IJCB) 2026**.

---

## Overview

Deep fingerprint PAD models remain sensitive to **nuisance geometric variations**
(in-plane rotation from unconstrained finger placement) and tend to **dilute
locally-confined spoof evidence** into a global descriptor. GARP-PAD tackles both by
design, in a model of only **≈ 2.6 × 10⁵ parameters (≈ 1 MB)** that runs in
**7.8 ms per fingerprint** on a single NVIDIA RTX 2080 Ti:

1. **Rotation-equivariant encoder** — an E(2)-steerable convolutional network with
   **C8** rotational equivariance (`e2cnn`) produces rotation-invariant patch
   descriptors *by construction*, rather than through data augmentation.
2. **Gated-attention MIL aggregator** — a multi-head gated-attention Multiple
   Instance Learning head (**H = 8**) weights each patch by its contribution to the
   decision, addressing the spatially non-uniform nature of spoof artifacts.
3. **Coherence-driven patch sampling** — patches are drawn only from a
   structure-tensor ROI covering stable ridge regions.

Under the LivDet 2021/2023/2025 **intra-sensor** protocol, GARP-PAD attains the lowest
Overall ACE on both Green Bit (**5.51%**) and Dermalog (**5.30%**), with markedly
balanced APCER/BPCER, despite being roughly an order of magnitude smaller than typical
PAD backbones.

---

## Architecture

```
Fingerprint image (grayscale)
        │
        ▼
Local contrast normalization  +  coherence-based ROI (structure tensor, top 15%)
        │
        ▼
Patch extraction  (S = 32 × 32, stride 8, keep patches ≥ 80% inside ROI)
        │
        ▼
C8 rotation-equivariant encoder  (e2cnn: stem + 2 depthwise-separable stages)
        │
        ▼
Norm-pooling over orientations  +  global average pooling  →  descriptor  hᵢ ∈ ℝ⁶⁴
        │
        ▼
Gated multi-head attention MIL  (H = 8)  →  bag embedding  z ∈ ℝ⁶⁴
        │
        ▼
Linear classifier  →  bona fide vs. attack
```

Training objective: **binary Focal Loss** (α = 0.75, γ = 2.0).

---

## Repository structure

```
GARP-PAD/
├── datasets/
│   └── fingerprint_dataset.py     # image-list dataset; label inferred from path
├── preprocessing/
│   └── preprocessor.py            # local contrast norm. + coherence ROI + patches
├── models/
│   ├── encoder_r2.py              # C8 rotation-equivariant encoder (main model)
│   ├── encoder_r2so2.py           # continuous SO(2)-steerable encoder (experimental)
│   ├── encoder_baseline.py        # non-equivariant CNN encoder (ablation baseline)
│   ├── aggregator.py              # gated-attention MIL + mean-pooling aggregators
│   └── garp_pad.py                # full model (encoder + aggregator + classifier)
├── losses/
│   └── focal_loss.py              # binary Focal Loss
├── scripts/
│   ├── train.py                   # training loop
│   └── eval.py                    # evaluation (APCER / BPCER / ACE)
├── checkpoints/                   # pretrained full-model weights (greenbit / dermalog)
├── data/                          # image-list files + a sample image
├── main.py                        # CLI entry point (train / eval)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

---

## Requirements

* Python 3.12
* `e2cnn==0.2.3`
* `kornia==0.8.2`
* `Pillow==12.2.0`
* `torch==2.11.0` (+cu128 build for GPU)
* `torchvision==0.26.0` (+cu128 build for GPU)
* `tqdm==4.67.3`

> ⚠️ `requirements.txt` pins the CUDA 12.8 (`+cu128`) PyTorch build. Install the wheel
> matching your own CUDA toolkit, or the CPU-only build, if it differs.

---

## Installation

```bash
git clone https://github.com/carsim97/GARP-PAD.git
cd GARP-PAD
pip install -r requirements.txt
```

---

## Data preparation

Each `data/*.txt` file is a plain list with **one image path per line**. Labels are
inferred from the path: a path containing the substring `live` is treated as a
**bona fide** sample (label `1`), any other path as an **attack** (label `0`).

```
/path/to/live/subject_01.png
/path/to/fake/gelatine_03.png
...
```

A minimal example (`data/data.txt` → `data/test.png`) is included so the pipeline can
be run end-to-end without the full datasets. The **LivDet 2021 / 2023 / 2025** datasets
are not redistributed here and must be obtained from the official
[LivDet](https://livdet.diee.unica.it/) competition organizers; build your own list
files (bona fide paths containing `live`) to reproduce the paper protocol.

---

## Usage

The CLI has two sub-commands. `--data_file` is resolved relative to `data/` and
`--checkpoint_file` relative to `checkpoints/`.

### Training

```bash
python main.py train \
  --data_file train_greenbit.txt \
  --checkpoint_file greenbit.pth \
  --epochs 50 --batch_size 128 --lr 1e-3 --num_patches 2
```

### Evaluation

```bash
python main.py eval \
  --data_file test_greenbit_2023.txt \
  --checkpoint_file greenbit.pth
```

Evaluation reports Accuracy, APCER, BPCER and ACE for the given test list. Pretrained
full-model checkpoints for the Green Bit and Dermalog sensors are provided under
`checkpoints/`.

---

## Ablation study

The component swaps analyzed in the paper (Table 4) are exposed as command-line flags —
no code editing or forked training loop is required. Everything else (data splits, patch
sampling, optimizer, seed) is held fixed.

| Flag           | Options                 | Default    | Meaning                                             |
|----------------|-------------------------|------------|-----------------------------------------------------|
| `--invariant`  | `normpool` \| `maxpool` | `normpool` | invariant descriptor: norm-pool vs. group max-pool  |
| `--aggregator` | `gated` \| `mean`       | `gated`    | MIL aggregator: gated attention vs. mean pooling    |

```bash
# Orientation group max-pooling instead of norm-pooling
python main.py train --data_file train_greenbit.txt \
  --checkpoint_file greenbit_maxpool.pth --invariant maxpool

# Unweighted mean pooling instead of gated attention
python main.py train --data_file train_greenbit.txt \
  --checkpoint_file greenbit_mean.pth --aggregator mean
```

The **encoder** ablation (rotation order / equivariance) is provided as separate encoder
modules in `models/`: the C8 model (`encoder_r2.py`), a continuous SO(2)-steerable
variant (`encoder_r2so2.py`), and a non-equivariant CNN baseline (`encoder_baseline.py`).

For ROI-robustness studies, `eval` additionally accepts `--roi_percentile`,
`--mask_ratio` and `--max_eval_patches`; their defaults reproduce the paper pipeline
exactly (sweeping them keeps Overall ACE within [5.29, 5.45]).

### Component ablation (Overall ACE, %)

Intra-sensor ACE averaged over LivDet 2021/2023/2025 (lower is better); Overall =
mean(Green Bit, Dermalog). Reproduced from Table 4 of the paper.

| Configuration                       | Green Bit | Dermalog | Overall |
|-------------------------------------|:---------:|:--------:|:-------:|
| **GARP-PAD (full)**                 | **5.51**  | **5.30** | **5.40**|
| Orientation max-pooling             |   5.19    |   6.43   |  5.81   |
| Mean pooling (no attention)         |   5.77    |   7.34   |  6.56   |
| Non-equivariant encoder (baseline)  |   8.72    |   7.56   |  8.14   |

Removing the equivariant encoder is the single most damaging change; the full paper
additionally studies the number of training patches, patch size, and the rotation
order (C4/C8/C16/C32).

---

## Metrics

Following **ISO/IEC 30107-3**:

* **APCER** — Attack Presentation Classification Error Rate
* **BPCER** — Bona Fide Presentation Classification Error Rate
* **ACE** — Average Classification Error

$$
\mathrm{ACE} = \frac{\mathrm{APCER} + \mathrm{BPCER}}{2}
$$

APCER and BPCER are always reported separately; ACE is used only as a single-number
summary.

---

## Experimental setup

* **Datasets:** LivDet — trained on the combined 2021 + 2023 train sets, evaluated on
  the 2021 / 2023 / 2025 test sets; intra-sensor results reported for Green Bit and
  Dermalog (cross-sensor results are also reported in the paper).
* **Patches:** S = 32; N = 2 random ROI patches per fingerprint at training, all
  stride-8 ROI patches at inference.
* **Optimizer:** Adam (β₁ = 0.9, β₂ = 0.999), cosine annealing from 1e-3, 50 epochs,
  batch size 128, on a single NVIDIA RTX 2080 Ti.
* **Loss:** binary Focal Loss (α = 0.75, γ = 2.0).
* **Augmentation:** constrained ±15° rotation only (no color jitter / noise).

---

## Citation

If you use this code or the pretrained models, please cite:

```bibtex
@inproceedings{carta2026garppad,
  title     = {{GARP-PAD}: A Gated Attention Rotation-equivariant Patch-based
               Architecture for Fingerprint Presentation Attack Detection},
  author    = {Carta, Simone and Casula, Roberto and Marcialis, Gian Luca},
  booktitle = {IEEE International Joint Conference on Biometrics (IJCB)},
  year      = {2026}
}
```

---

## License

Released under the [MIT License](LICENSE) for research purposes.

## Acknowledgements

Developed at the [PRA Lab](https://pralab.diee.unica.it/), Department of Electrical and
Electronic Engineering, University of Cagliari.
