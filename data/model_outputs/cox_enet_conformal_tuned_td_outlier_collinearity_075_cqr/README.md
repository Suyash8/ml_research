# Tuned Cox ENet + CQR conformal intervals

## Method
Conformal intervals use Conformalized Quantile Regression (CQR).
Reference: Romano, Patterson & Candès, NeurIPS 2019, arXiv:1905.03222.

CQR replaces the previous IsotonicRegression + log1p/expm1 approach,
which produced artificially wide intervals due to exponential back-
transformation of log-space residuals.

## Pipeline
Input               : /home/illionar/Projects/ml_research/data/preprocessed_cleaned/patient_multiomic_cleaned.parquet
Collinearity thresh : 0.75 (abs Pearson)
Best Cox alpha      : 3.0
Best Cox l1_ratio   : 0.7
CV best mean c-idx  : 0.7129
Test c-index        : 0.7203

## CQR month-interval results
Status              : ok
Coverage (events)   : 0.8470
qhat (months)       : 0.37
Mean width (months) : 52.2

## Collinearity filter
Clinical : 6 -> 6
Expression: 500 -> 57
Total dropped: 443

## Output files
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