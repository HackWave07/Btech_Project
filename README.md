# Severity and Progression Conditioned Lesion-Aware GAN (SPC-GAN)

This repository contains the complete implementation of the SPC-GAN pipeline for early tomato leaf disease detection.

## Structure
- `config/`: Hyperparameters and configuration.
- `src/`: Core implementation (models, data loaders, training loops, losses).
- `scripts/`: Utility scripts for preparation and generation.
- `app/`: Streamlit web application for inference.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
```bash
python main.py --mode prepare
```

### 3. Demo Mode (Quick Run)
```bash
python main.py --mode demo
```

### 4. Train Baseline Classifier
```bash
python main.py --mode train_classifier
```

### 5. Train SPC-GAN
```bash
python main.py --mode train_gan
```

### 6. Evaluate Model
```bash
python main.py --mode evaluate
```

### 7. Run Streamlit App
```bash
streamlit run app/app.py
```
