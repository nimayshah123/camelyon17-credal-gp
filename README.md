# Credal GPs and the Expansion Function — Camelyon17-WILDS Case Study

## Overview

This repository contains the full experimental pipeline for the paper:
**"Credal GPs and the Expansion Function: Dissolving the Generalization-Calibration Tradeoff in OOD Medical Imaging"**

We apply credal Gaussian Processes to the Camelyon17-WILDS dataset — 455,954 pathology image patches from 5 Dutch hospitals — demonstrating that the generalization-calibration tradeoff in OOD settings is a representational artifact dissolved by credal sets.

### Key Concepts

- **Expansion function**: `e_φ(z) = σ²_between(z) / σ²_within(z)` — measures how much a feature shifts across hospital domains.
- **Credal GP**: a set of diverse GP kernels `{k_θ : θ ∈ Θ}`. Automatically widens OOD because diverse kernels disagree where data is absent.
- **Closed loop**: expansion function identifies which new hospital domain to collect → adding it narrows credal width exactly there.

### Completed Precedents

This pipeline extends successful case studies on:
1. UCI Wine (3 cultivars as domains, credal GP r=0.744)
2. Breast Cancer Wisconsin (4 cohorts, 30 features, r=0.976)

---

## Setup

```bash
pip install -r requirements.txt
```

For WILDS dataset (auto-downloaded on first run):
```bash
pip install wilds
```

---

## Running the Pipeline

### Full pipeline (single command):
```bash
python run_all.py
```

### Fast development mode (500 samples/hospital, ~minutes on CPU):
```bash
python run_all.py --fast-dev
```

### Individual steps:
```bash
python scripts/01_extract_features.py    # Extract ResNet50 features + PCA
python scripts/02_expansion_analysis.py  # Compute expansion function
python scripts/03_credal_gp_analysis.py  # Run credal GP
python scripts/04_baselines.py           # Train/evaluate baselines
python scripts/05_metrics_comparison.py  # Full metrics table
python scripts/06_domain_selection.py    # Hospital ranking + greedy selection
```

---

## File Structure

```
camelyon17-credal-gp/
├── requirements.txt
├── README.md
├── run_all.py                    # Master script
├── config.py                     # All hyperparameters and paths
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Load Camelyon17-WILDS, split by hospital
│   ├── feature_extractor.py      # ResNet50 feature extraction + PCA
│   ├── expansion.py              # Expansion function computation
│   ├── credal_gp.py              # Credal GP: kernel set, posterior, width
│   ├── baselines.py              # ERM, MC Dropout, Deep Ensemble
│   ├── metrics.py                # ECE, NLL, Brier, AUROC, FPR@95
│   ├── domain_selection.py       # Greedy hospital ranking
│   └── visualize.py              # All 3 figures
├── scripts/
│   ├── 01_extract_features.py
│   ├── 02_expansion_analysis.py
│   ├── 03_credal_gp_analysis.py
│   ├── 04_baselines.py
│   ├── 05_metrics_comparison.py
│   └── 06_domain_selection.py
├── features/                     # Auto-created: .npy feature arrays
└── outputs/                      # Auto-created: figures (PNG + PDF)
```

---

## Outputs

All figures are saved to `outputs/` as both PNG (150 dpi) and PDF:

- `figure1_domain_profiles.png/pdf` — Domain clusters, expansion bar chart, per-hospital heatmap
- `figure2_credal_gp.png/pdf` — Single GP vs credal GP uncertainty bands + correlation
- `figure3_domain_selection.png/pdf` — Hospital ranking, before/after width, metrics table, greedy curve

A metrics summary table is printed to console at the end of `run_all.py`.

---

## Dataset

[Camelyon17-WILDS](https://wilds.stanford.edu/datasets/#camelyon17) — 455,954 pathology patches (96×96 px) from 5 Dutch hospitals:
- Hospital 0: Radboud UMC
- Hospital 1: CWZ
- Hospital 2: UMCU
- Hospital 3: RST (held out as OOD test domain)
- Hospital 4: LPON

---

## Citation

If you use this code, please cite:
```
[Paper citation to be added upon publication]
```
