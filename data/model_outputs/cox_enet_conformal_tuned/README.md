# Tuned Cox ENet + Conformal

Input: /home/illionar/Projects/ml_research/data/preprocessed_cleaned/patient_multiomic_cleaned.parquet
Best alpha: 3.0
Best l1_ratio: 0.7
CV best mean c-index: 0.7285
Test c-index: 0.7456
Conformal 90% coverage on uncensored test events: 0.8307

Artifacts:
- hyperparameter_cv_results.csv
- tuned_model_metrics.json
- tuned_model_predictions.csv
- tuned_model_coefficients.csv
- final_locked_model.pkl