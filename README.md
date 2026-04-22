# GARP-PAD: Rotation-Equivariant Fingerprint Presentation Attack Detection

Official PyTorch implementation of **GARP-PAD**, a rotation-equivariant, patch-based deep learning framework for **Fingerprint Presentation Attack Detection (PAD)**.

> 📄 Based on the IJCB 2026 submission:
> *“GARP-PAD: A Gated Attention Rotation-equivariant Patch-based Network for Robust Fingerprint Presentation Attack Detection”* 

---

## 🔍 Overview

Fingerprint PAD aims to distinguish between:

* **Bona fide samples** (real fingers)
* **Presentation attacks** (spoofs made of silicone, latex, screens, etc.)

Traditional CNN-based methods struggle with:

* ❌ sensitivity to **rotation / finger placement**
* ❌ dilution of **local spoof evidence**

---

## 💡 Key Idea

GARP-PAD models a fingerprint as a **set of local patches** and introduces:

### 🧠 1. Rotation-Equivariant Encoder

* Based on **E(2)-steerable convolutions**
* Guarantees equivariance to in-plane rotations
* Eliminates need for heavy data augmentation 

### 🧩 2. Patch-Based Representation

* Fingerprint → set of local micro-structures
* Focus on **intrinsic spoof cues**, not global patterns 

### 🎯 3. Gated Attention Aggregation (MIL)

* Learns which patches matter
* Handles **spatially non-uniform spoof artifacts** 

### ⚙️ 4. Coherence-Driven Patch Sampling

* Uses gradient structure tensor
* Extracts only **informative ridge regions** 

---

## 🏗️ Architecture

Pipeline:

```
Image
  ↓
ROI extraction (coherence-based)
  ↓
Patch extraction (32×32)
  ↓
Rotation-equivariant encoder (E(2))
  ↓
Invariant mapping (norm pooling)
  ↓
Gated multi-head attention
  ↓
Binary classification
```

---

## 📁 Repository Structure

```
GARP-PAD/
│
├── datasets/             # Dataset loader
├── preprocessing/        # ROI + patch extraction
├── models/               # Encoders + GARP-PAD
├── training/             # Train / eval logic
├── scripts/              # CLI scripts
├── utils/                # Config, metrics
│
├── main.py               # Entry point
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/carsim97/GARP-PAD.git
cd GARP-PAD

pip install -r requirements.txt
```

### 🔥 PyTorch (GPU)

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## 🚀 Usage

### 🏋️ Training

```bash
python main.py train --data_file <txt_file> --checkpoint_file <pth_file>
```

---

### 📊 Evaluation

```bash
python main.py eval --data_file <txt_file> --checkpoint_file <pth_file>
```

---

## 📊 Metrics

Following ISO/IEC 30107-3:

* **APCER**: Attack Presentation Classification Error Rate
* **BPCER**: Bona Fide Presentation Classification Error Rate
* **ACE**: Average Classification Error

$$
ACE = \frac{APCER + BPCER}{2}
$$

---

## 🧪 Experimental Setup

* Datasets: **LivDet 2021, 2023, 2025**
* Training:

  * Adam optimizer
  * LR = 1e-3 (cosine annealing)
  * Batch size = 128
  * 50 epochs
* Loss:

  * Focal Loss (α=0.75, γ=2.0) 

---

## 📈 Key Results

### ✅ Strengths

* Robust to **rotation and finger placement**
* Strong **intra-sensor performance**
* Stable across dataset evolution 

### ⚠️ Limitations

* Performance drops in **cross-sensor scenarios**
* Sensitive to **sensor-specific noise/physics** 

---

## 🧠 Insights

* Rotation equivariance is **more effective than augmentation**
* Spoof artifacts are **local and heterogeneous**
* Attention improves robustness by focusing on **salient regions**

---

## 📌 TODO / Future Work

* [ ] Domain adaptation for cross-sensor robustness
* [ ] Real-time deployment optimization

---


## ⚠️ Disclaimer

This repository is for **research purposes only**.
The associated paper is currently under review.

---

## 🙌 Acknowledgements

* LivDet benchmark series
* E(2)-CNN framework
* PyTorch ecosystem

---

## ⭐ If you find this useful

Consider starring ⭐ the repo!
