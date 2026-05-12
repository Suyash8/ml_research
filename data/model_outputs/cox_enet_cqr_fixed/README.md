# Cox ENet + CQR — FIXED wide-interval bug

## Root cause of old wide intervals
The previous pipeline used IsotonicRegression + log1p/expm1:
  lo = expm1(yhat − qhat),  hi = expm1(yhat + qhat)
Because expm1 is convex, a qhat of ~1.5 in log-space maps
asymmetrically: hi can be 30–100× larger than lo.

## Fix: CQR on raw month scale
CQR (Romano, Patterson & Candès, NeurIPS 2019) operates in months:
  lo = q_lo(X) − qhat_months,  hi = q_hi(X) + qhat_months
Both bounds shift by the SAME absolute number of months.
qhat is now in months (not log-months), so no explosion occurs.

## Results
Test c-index      : 0.7203
CV mean c-index   : 0.7129
CQR status        : ok
qhat              : 1.83 months
Coverage (events) : 0.8907
Mean width (all)  : 52.4 months

## Tightening levers (if intervals still too wide)
1. Increase CQR_MIN_TRAIN_EVENTS (requires more data).
2. Tune CQR_GBR_PARAMS: higher n_estimators, lower max_depth.
3. Set MAX_INTERVAL_WIDTH_MONTHS = e.g. 36 for a hard cap.
4. Accept that interval width reflects genuine uncertainty;
   with low event rates there is no algorithmic fix.

## Files
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