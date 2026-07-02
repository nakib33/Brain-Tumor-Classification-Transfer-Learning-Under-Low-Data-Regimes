# 🧠 Brain Tumor Classification — Transfer Learning Under Low-Data Regimes

### PyTorch Implementation | EfficientNet-B0 | Transfer Efficiency Index (TEI)

**Paper:** *Cross-Domain Transfer Learning for Brain Tumor Classification Under Limited MRI Data Regimes*

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNet--B0-EE4C2C)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20Complete-brightgreen)

---

## 📋 Overview

This project investigates **how much labeled MRI data is actually needed** to build a clinically useful brain tumor classifier. Using **EfficientNet-B0** as the backbone, it systematically compares **ImageNet-pretrained transfer learning** against **training from scratch** across six progressively larger training-data fractions (5% → 100%).

The goal is to answer a practical question radiologists and ML engineers care about: *"If I only have a small labeled dataset, is transfer learning worth it — and at what point does the extra data stop mattering?"*

To answer this rigorously, the notebook introduces a custom metric — the **Transfer Efficiency Index (TEI)** — and a clinically motivated threshold — the **Minimum Viable Data Threshold (MVDT)** — and pairs the quantitative results with **Grad-CAM / Grad-CAM++ visual explanations** so the model's decisions can be inspected, not just trusted.

---

## 🎯 Key Features

- **Two training regimes compared head-to-head:** ImageNet-pretrained fine-tuning vs. random-initialization ("from scratch") training
- **Low-data simulation:** stratified sampling at **5%, 10%, 20%, 30%, 50%, and 100%** of the training set
- **3-stage gradual unfreezing:** classifier head → top 2 backbone blocks → full backbone, each with its own learning rate
- **Focal Loss + Label Smoothing** to handle hard/ambiguous MRI slices better than plain cross-entropy
- **Cosine Annealing with Warm Restarts** learning-rate scheduling
- **Multi-seed evaluation** (3 seeds per configuration) for statistically meaningful accuracy/F1 estimates
- **Power-cut-safe checkpointing** — training can resume mid-epoch after an interruption (e.g., a Colab disconnect) without losing progress
- **Transfer Efficiency Index (TEI):** a novel metric quantifying *how much value* pretraining adds, normalized by training-set size
- **Minimum Viable Data Threshold (MVDT):** identifies the smallest data fraction at which the model reaches a clinically acceptable Macro F1 (≥ 0.85)
- **Grad-CAM / Grad-CAM++ explainability:** visualizes what the model "looks at" when making a prediction, including how attention quality evolves with more data and how it differs between pretrained and scratch models
- **Publication-quality figures:** every plot is saved as a 300 DPI PNG, ready to drop into a paper or report

---

## 🗂️ Dataset

**Source:** [Brain Tumor MRI Dataset — Masoud Nickparvar (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

| Property | Value |
|---|---|
| Total images | 7,200 (balanced) |
| Classes | Glioma, Meningioma, Pituitary Tumor, No Tumor |
| Format | Pre-split `Training/` and `Testing/` folders, one subfolder per class |
| Image type | Grayscale brain MRI scans (converted to RGB for the pretrained backbone) |

### Downloading the data

```bash
# Option A — kagglehub (recommended, used in the notebook)
pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('masoudnickparvar/brain-tumor-mri-dataset'))"

# Option B — Kaggle CLI
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d ./data/brain_tumor_mri/
```

Expected folder structure:

```
data/brain_tumor_mri/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Update the `DATA_DIR` variable in the **Configuration** section of the notebook to point to this folder.

---

## 🏗️ Model Architecture

**Backbone:** `EfficientNet-B0` (via `timm`), pretrained on ImageNet or randomly initialized.

**Classification head:**

```
GlobalAvgPool → Dropout(0.3) → Linear(1280, 512) → SiLU → BatchNorm1d(512)
             → Dropout(0.3) → Linear(512, 4)
```

**Training strategy — 3-stage gradual unfreezing:**

| Stage | Epochs | What's trainable | Learning rate |
|---|---|---|---|
| 1 | 1–10 | Classifier head only | 1e-3 |
| 2 | 11–25 | Head + top 2 backbone blocks | 1e-4 |
| 3 | 26+ | Full model | 1e-5 |

**Loss function:** Focal Loss (γ = 2.0, α = 0.25) with label smoothing (0.1), designed to down-weight easy examples and focus training on hard/ambiguous scans.

**Regularization / augmentation:** random crop, horizontal/vertical flip, rotation, color jitter, affine transforms, random erasing (scaled down for low-data fractions to avoid over-regularizing tiny datasets), gradient clipping, and early stopping (patience = 12 epochs).

---

## 📐 Custom Metrics

### Transfer Efficiency Index (TEI)

Quantifies how much benefit pretraining gives, adjusted for how much data is available:

```
TEI = (Accuracy_finetuned − Accuracy_scratch) / log10(N_samples)
```

| TEI range | Interpretation |
|---|---|
| > 0.15 | Excellent transfer |
| 0.10 – 0.15 | Good transfer |
| 0.05 – 0.10 | Moderate transfer |
| 0 – 0.05 | Weak transfer |
| < 0 | Negative transfer (scratch outperforms pretrained) |

### Minimum Viable Data Threshold (MVDT)

The smallest training-data fraction at which the fine-tuned model reaches a **Macro F1 ≥ 0.85** — used as a proxy for "clinically usable" performance, helping estimate how many labeled scans a real deployment would realistically need.

---

## 📊 Results & Figures

> All figures below are generated by the notebook at **300 DPI** and saved to `results/figures/`. Copies referenced here live in the `images/` folder — add your generated PNGs there with the filenames below (they match the names the notebook already saves them as) to have them render automatically in this README.

### Dataset exploration

| Class distribution | Sample MRI images |
|---|---|
| ![Class Distribution](images/fig0a_class_distribution.png) | ![Sample Images](images/fig0b_sample_images.png) |

| Pixel intensity distribution | Experiment design (samples per fraction) |
|---|---|
| ![Pixel Intensity](images/fig0c_pixel_intensity.png) | ![Fraction Design](images/fig0d_fraction_design.png) |

### Main results

**Figure 1 — Accuracy, F1, and TEI vs. training-data fraction** (fine-tuned vs. from-scratch):

![Main Results](images/fig1_main_results.png)

**Figure 2 — Confusion matrices** at key data fractions (10% / 50% / 100%):

![Confusion Matrices](images/fig2_confusion_matrices.png)

**Figure 3 — Training curves** (loss & accuracy) per data fraction:

![Training Curves](images/fig3_training_curves.png)

**Figure 4 — Per-class F1 heatmap** across all data fractions:

![Per-Class F1 Heatmap](images/fig4_perclass_f1_heatmap.png)

**Figure 5 — Learning-rate schedule** (Cosine Annealing with Warm Restarts, 100% data):

![LR Schedule](images/fig5_lr_schedule.png)

### Explainability (Grad-CAM)

| Per-class Grad-CAM grid | Misclassified samples |
|---|---|
| ![GradCAM Grid](images/gradcam_grid.png) | ![Misclassified GradCAM](images/gradcam_misclassified.png) |

| Attention evolution across data fractions | Pretrained vs. scratch attention |
|---|---|
| ![Attention Evolution](images/gradcam_attention_evolution.png) | ![Pretrained vs Scratch](images/gradcam_pretrained_vs_scratch.png) |

> 💡 **Tip:** the notebook auto-names every saved figure exactly as listed above — after running it end-to-end, just copy `results/figures/*.png` into this repo's `images/` folder and every image link in this README will resolve automatically.

---

## 🧪 Output Files

All experiment artifacts are saved under `results/`:

| File | Description |
|---|---|
| `results/tei_results.csv` | TEI + accuracy/F1 pivot table across all fractions |
| `results/all_results.csv` | Raw per-run results (accuracy, F1, seed, mode, fraction) |
| `results/tei_summary.csv` | Final paper-ready TEI summary table |
| `results/results_summary.json` | Full results dump (excluding large arrays) |
| `results/models/` | Best model weights per fraction/mode |
| `results/epoch_ckpts/` | Per-epoch checkpoints (for crash-safe resuming) |
| `results/figures/` | All publication-quality PNGs (300 DPI) |

---

## ⚙️ Installation

```bash
git clone <your-repo-url>
cd brain-tumor-transfer-learning

pip install torch torchvision timm scikit-learn matplotlib seaborn pandas tqdm kagglehub opencv-python
```

**Requirements:**

- Python 3.9+
- PyTorch (GPU strongly recommended — a full run trains 36 models: 6 fractions × 2 modes × 3 seeds)
- `timm` (EfficientNet-B0 backbone)
- `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `tqdm`
- `opencv-python` (for Grad-CAM heatmap overlays)
- `kagglehub` (optional, for one-line dataset download)

---

## 🚀 Usage

1. **Download the dataset** and set `DATA_DIR` in the Configuration cell.
2. **Run the notebook top to bottom**, or run each `run_fraction(...)` cell independently (5%, 10%, 20%, 30%, 50%, 100%) — training is checkpointed every epoch, so if a session disconnects, simply re-run the same cell and it resumes automatically.
3. **Collect results** — the notebook aggregates all runs, computes TEI, and generates every figure listed above.
4. **Run the Grad-CAM sections** at the end to generate explainability visualizations for the saved models.

```python
# Example: run a single data-fraction experiment
results_10pct = run_fraction(0.10)
```

---

## 🔬 Methodology Summary

1. Load the Nickparvar Brain Tumor MRI dataset (7,200 balanced images, 4 classes).
2. Create a stratified 85/15 train/validation split from the `Training/` folder; `Testing/` is held out as the final test set.
3. For each data fraction (5–100%), stratified-subsample the training set.
4. For each fraction, train **two models** — one initialized from ImageNet weights, one from scratch — across **3 random seeds** each, using 3-stage gradual unfreezing and Focal Loss.
5. Evaluate every model on the untouched test set (accuracy, macro F1, per-class F1, confusion matrix).
6. Compute **TEI** to quantify the value of pretraining at each data scale, and determine the **MVDT**.
7. Generate Grad-CAM visualizations to interpret model attention and compare pretrained vs. scratch localization quality.

---

## 📁 Project Structure

```
.
├── Brain_Tumor_Transfer_Learning_V2.ipynb   # Main notebook (this project)
├── data/
│   └── brain_tumor_mri/                     # Dataset (Training/ + Testing/)
├── images/                                  # Figures used in this README
├── results/
│   ├── figures/                             # Generated plots (300 DPI PNGs)
│   ├── models/                              # Best model weights
│   ├── epoch_ckpts/                         # Resumable per-epoch checkpoints
│   ├── tei_results.csv
│   ├── all_results.csv
│   ├── tei_summary.csv
│   └── results_summary.json
└── README.md
```

---

## ✅ Key Improvements in V2

- New dataset: 7,200 perfectly balanced images (vs. 3,264 imbalanced in V1)
- Focal Loss replaces plain Cross-Entropy Loss
- 3-stage gradual unfreezing (head → top 2 blocks → full model)
- Cosine Annealing with Warm Restarts scheduler
- Gradient clipping for training stability
- Power-cut-safe, resumable checkpointing
- Grad-CAM / Grad-CAM++ explainability suite
- Publication-quality (300 DPI) figure generation throughout

---

## 📖 Citation

If you use this work, please cite the associated paper:

> *Cross-Domain Transfer Learning for Brain Tumor Classification Under Limited MRI Data Regimes*

and the dataset:

> Nickparvar, M. *Brain Tumor MRI Dataset.* Kaggle.

---

## 📝 License

This project is released under the MIT License. The underlying dataset follows its own license terms on Kaggle — please review those before any clinical or commercial use.

---

## ⚠️ Disclaimer

This model is a **research prototype** built for academic experimentation on transfer learning and low-data regimes. It is **not validated for clinical use** and must not be used for real diagnostic decisions without proper regulatory approval, larger-scale validation, and clinical oversight.
