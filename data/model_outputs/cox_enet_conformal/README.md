# Cox PH (Elastic Net) + Conformal Uncertainty

Input: /home/illionar/Projects/ml_research/data/preprocessed_cleaned/patient_multiomic_cleaned.parquet
N=1620 (train=972, calibration=324, test=324)

## Cox Elastic Net
- alpha=0.8
- l1_ratio=0.3
- c-index train=0.7550
- c-index calibration=0.7373
- c-index test=0.7438

## Conformal Uncertainty (90% interval)
- qhat (abs residual on log-months)=1.2816
- empirical coverage on test events=0.8519
- mean interval width on test=86.60 months

## Files
- cox_enet_conformal_metrics.json
- cox_enet_conformal_predictions.csv
- cox_enet_coefficients.csv