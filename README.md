# ML Research: Cox Elastic-Net Survival Pipeline

This repository contains a modular, end-to-end machine learning pipeline for survival analysis. At its core, the project utilizes a custom **Cox Proportional Hazards model with Elastic-Net regularization** (`CoxElasticNet`), capable of handling multi-omic datasets (combining clinical features and gene expression profiles). 

The pipeline includes built-in collinearity filtering, automated hyperparameter tuning (via nested cross-validation), isotonic calibration for specific time horizons, and Monte Carlo simulations for probabilistic survival estimates.

## Repository Structure

The codebase has been refactored from legacy monolithic scripts into a clean, reusable python package.

```
ml_research/
├── src/                      # Core package containing reusable modules
│   ├── data/                 # Data loading and splitting logic
│   ├── features/             # Collinearity filtering and PCA preprocessing
│   ├── models/               # Custom CoxElasticNet and isotonic calibration
│   ├── metrics/              # Survival metrics (C-index, Brier, AUROC) and Explainability (XAI)
│   ├── training/             # Cross-validation and Monte Carlo simulations
│   └── utils/                # Configuration constants, I/O, and plotting tools
├── scripts/                  # Thin CLI entrypoints for running the pipeline
│   ├── run_cox_enet_pipeline.py  # End-to-end training and prediction
│   ├── run_explainability.py     # Generates patient-level explanations from a locked model
│   ├── plot_results.py           # Generates plots (MAPE, heatmaps, waterfall charts)
│   └── data_prep/                # Utilities for converting and cleaning raw data
├── archive/                  # Legacy v2-v4 pipeline scripts (preserved via git mv)
├── notebooks/                # Exploratory Jupyter notebooks
└── setup.py                  # Package definition for local installation
```

## Installation

To ensure that Python can resolve internal module imports (e.g., `from ml_research.src...`), install the package locally in editable mode:

```bash
# From the root of the repository
pip install -e .
```

## Usage

### 1. Model Training & Prediction
Run the main pipeline to fit the model, apply collinearity filters, tune hyperparameters, and generate Monte Carlo survival predictions.

```bash
python scripts/run_cox_enet_pipeline.py --input data/preprocessed_cleaned/patient_multiomic_cleaned.parquet
```
*Outputs will be saved to `results/cox_enet_calibrated_mc_outputs_v5/` and the locked model artifact to `model_weights/final_locked_model.pkl`.*

### 2. Explainability (XAI)
Once the model is trained, you can run the explainability script to extract global feature importances, PCA back-projections, and patient-level risk contributions.

```bash
python scripts/run_explainability.py
```
*Outputs will be saved to `results/explainability/`.*

### 3. Plotting Results
Generate visual reports including MAPE scatter plots, feature heatmaps, and local patient waterfall charts.

```bash
python scripts/plot_results.py
```
*Plots will be saved to `results/plots/`.*

## Data Preparation

If you are starting from raw data, the `scripts/data_prep/` directory contains tools for parsing, cleaning, and converting your raw files into the expected `.parquet` format used by the pipeline:
- `convert_to_parquet.py`
- `create_cleaned_datasets.py`
- `parse_data.py`
