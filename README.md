# SPC-GAN: Tomato Leaf Disease Detection & Severity Estimation

> **B.Tech Research Project** — Severity-conditioned Progressive Conditional GAN for automated plant disease detection using the PlantVillage dataset.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)

---

## 📖 Project Overview

SPC-GAN combines a **Generative Adversarial Network** conditioned on disease class and severity level with a **ResNet-18 dual-head classifier** to:

1. **Detect** tomato leaf diseases across 10 categories.  
2. **Estimate severity** (mild / moderate / severe) from a single leaf image.  
3. **Generate** realistic synthetic augmentation images to address class imbalance.  
4. **Explain** predictions visually via Grad-CAM heatmaps.

### Supported Classes

| # | Class | Type |
|---|-------|------|
| 1 | Tomato_Bacterial_spot | Disease |
| 2 | Tomato_Early_blight | Disease |
| 3 | Tomato_Late_blight | Disease |
| 4 | Tomato_Leaf_Mold | Disease |
| 5 | Tomato_Septoria_leaf_spot | Disease |
| 6 | Tomato_Spider_mites_Two_spotted_spider_mite | Pest |
| 7 | Tomato_Target_Spot | Disease |
| 8 | Tomato_Tomato_YellowLeafCurlVirus | Virus |
| 9 | Tomato_Tomato_mosaic_virus | Virus |
| 10 | Tomato_healthy | Healthy |

---

## 🏗️ Model Architecture

```
PlantVillage Images
       │
       ▼
┌─────────────────────────┐
│  ResNet-18 Backbone      │  ← Feature extractor
├─────────────────────────┤
│  Disease Head  (10-cls) │  ← CrossEntropyLoss
│  Severity Head  (3-cls) │  ← CrossEntropyLoss
└─────────────────────────┘
       │ Grad-CAM Attention
       ▼
┌─────────────────────────────────────────────────┐
│  SPC-GAN                                        │
│  ┌─────────────┐    ┌──────────────────────┐   │
│  │  Generator   │───▶│ Discriminator         │   │
│  │  z + class  │    │  real/fake + cls/sev  │   │
│  │  + severity │    └──────────────────────┘   │
│  └─────────────┘                               │
│        ↑ GradCAM Consistency Loss               │
└─────────────────────────────────────────────────┘
```

**Key innovations:**
- Severity-conditioned generation — output diversity controlled by disease severity label.
- Grad-CAM Consistency Loss — generator attention aligned with classifier's discriminative regions.
- Attention Regularisation — prevents attention map collapse.

---

## 📁 Project Structure

```
project-root/
├── app.py                  ← Streamlit web app (entry point)
├── main.py                 ← CLI pipeline runner
├── requirements.txt
├── packages.txt            ← Streamlit Cloud system packages
├── setup.sh
├── .gitignore
├── .streamlit/
│   └── config.toml         ← Theme & server settings
├── assets/
│   ├── images/
│   └── sample_outputs/     ← Training plots saved here
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/          ← GAN-generated images saved here
├── models/
│   ├── checkpoints/        ← .pth checkpoints saved here
│   └── trained/            ← TorchScript exports
├── src/
│   ├── config/config.py    ← Central Config class
│   ├── data/dataset.py     ← Dataset & DataLoader
│   ├── models/             ← Generator, Discriminator, Classifier
│   ├── training/           ← Training loops
│   ├── evaluation/         ← Metrics & confusion matrix
│   ├── inference/predict.py
│   ├── losses/losses.py    ← Custom loss functions
│   └── utils/              ← helpers, visualization, preprocessing
├── scripts/
│   ├── prepare_dataset.py
│   ├── generate_synthetic.py
│   └── export_model.py
└── notebooks/
    └── experimentation.ipynb
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended) or CPU

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/spc-gan-tomato.git
cd spc-gan-tomato

# 2. Create and activate virtual environment
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Dataset Setup
Download the [PlantVillage dataset](https://github.com/spMohanty/PlantVillage-Dataset) and place the tomato folders inside a `PlantVillage/` directory at the project root:

```
PlantVillage/
├── Tomato_Bacterial_spot/
├── Tomato_Early_blight/
...
└── Tomato_healthy/
```

---

## 🔬 Training Steps

### 1. Prepare the dataset

```bash
python main.py --mode prepare
```
Scans `PlantVillage/`, estimates severity labels via HSV analysis, and saves `dataset.csv`.

### 2. Train the baseline classifier

```bash
python main.py --mode train_classifier
```
Trains ResNet-18 with dual classification heads. Checkpoint saved to `models/checkpoints/baseline_classifier.pth`.

### 3. Train the SPC-GAN

```bash
python main.py --mode train_gan
```
Trains the generator and discriminator using Grad-CAM consistency loss.

### 4. Generate synthetic images

```bash
python main.py --mode generate --num_samples 200
```
Generates synthetic augmentation images into `data/synthetic/`.

### 5. Evaluate

```bash
python main.py --mode evaluate
```
Outputs accuracy, precision, recall, F1, and confusion matrix to `metrics/` and `assets/sample_outputs/`.

### 6. Quick demo (1 epoch each)

```bash
python main.py --mode demo
```

### 7. Export to TorchScript

```bash
python scripts/export_model.py \
  --checkpoint models/checkpoints/best_classifier.pth \
  --output     models/trained/classifier_scripted.pt
```

---

## 🌐 Streamlit App

### Run locally

```bash
streamlit run app.py
```

### App features
- **Sidebar navigation** – Diagnose / Model Info / About pages
- **Image upload** – JPG / PNG support
- **Prediction results** – Disease class + severity with confidence scores
- **Confidence charts** – Horizontal bar charts for all class and severity probabilities
- **Disease information panel** – Description, treatment, and severity notes
- **Grad-CAM heatmap** – Visual explanation of model decision

---

## ☁️ Streamlit Community Cloud Deployment

1. Push the project to a **public GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your repository, branch `main`, and set **Main file path** to `app.py`.
4. Streamlit Cloud automatically reads:
   - `requirements.txt` for Python packages
   - `packages.txt` for system packages (`libgl1-mesa-glx` for OpenCV)
   - `setup.sh` for additional setup
5. Click **Deploy**.

> **Note:** The app runs in UI-only mode if no model checkpoint is committed. To include a small demo model, commit `models/checkpoints/best_classifier.pth` after removing it from `.gitignore`.

---

## 📸 Screenshots

| Diagnosis Page | Grad-CAM |
|---|---|
| *(Upload a screenshot here)* | *(Upload a screenshot here)* |

---

## 📄 License

This project is for academic research purposes.  
Dataset credits: [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) (Hughes & Salathé, 2015).
