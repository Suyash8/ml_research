# Tuned Cox ENet + Outlier Filter + Collinearity Filter + Time-dependent Conformal

Input: /home/illionar/Projects/ml_research/data/preprocessed_cleaned/patient_multiomic_cleaned.parquet
Collinearity threshold (abs Pearson): 0.9
Best alpha: 3.0
Best l1_ratio: 0.7
CV best mean c-index: 0.7187
Test c-index: 0.7193
Month-interval conformal status: ok
Month-interval conformal coverage (event-only): 0.8361

Collinearity filter summary:
- Clinical columns: 6 -> 6
- Expression columns: 500 -> 234
- Total dropped: 266

Files:
- collinearity_dropped_features.csv
- collinearity_summary.json
- hyperparameter_cv_results.csv
- tuned_model_metrics.json
- tuned_model_predictions.csv
- tuned_model_coefficients.csv
- time_dependent_conformal_metrics.csv
- time_dependent_conformal_test_predictions.csv
- time_dependent_horizon_classification_metrics.csv
- consistency_checks.json
- final_locked_model.pkl