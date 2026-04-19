# Cox ENet + CQR conformal survival intervals

## Method summary
Conformal intervals use Conformalized Quantile Regression (CQR).
Reference: Romano, Patterson & Candès, NeurIPS 2019 (arXiv:1905.03222).

CQR fits two GradientBoostingRegressor quantile models at levels
α/2 = 0.05 and 1-α/2 = 0.95 on
event-only training rows (raw month scale, no log transform).
Conformal correction qhat is computed on event-only calibration rows.
Final interval: [q_lo(X) - qhat, q_hi(X) + qhat], clipped to [0, ∞).

## Coverage statement
~90% marginal coverage claimed for event-only test rows under
exchangeability. Coverage for censored patients is NOT claimed.

## Results
Input               : /home/illionar/Projects/ml_research/data/preprocessed_cleaned/patient_multiomic_cleaned.parquet
Collinearity thresh : 0.75
Best Cox alpha      : 3.0
Best Cox l1_ratio   : 0.7
CV mean c-index     : 0.7129
Test c-index        : 0.7203
CQR status          : ok
CQR qhat (months)   : 0.37
CQR coverage (ev.)  : 0.8470
Mean width (all te) : 52.2 months

## Wide interval diagnosis
If intervals are wide, likely causes (in order of probability):
  1. Low event rate -> few calibration events -> large qhat.
  2. High genuine variability in OS_MONTHS (e.g. from longitudinal
     model transformation) -> wide raw QR bands.
  3. Small sample size.
There is no algorithmic fix for (2) and (3) without more data.

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