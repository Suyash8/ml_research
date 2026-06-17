# Cox ENet + raw-month CQR + Monte Carlo

## What this pipeline reports
- risk score from the fitted Cox Elastic-Net model
- conformal interval on raw months
- Monte Carlo survival summaries at fixed horizons
- bounded RMST at 60 months

## Important interpretation note
The Monte Carlo block does not predict an exact survival lifespan.
It gives simulation-based summaries under the fitted Cox model and
the estimated Breslow baseline hazard.

The p90 simulated survival month is an upper-plausible time, not a
guarantee that the patient will survive that long.

## Results
Test c-index      : 0.7203
CV mean c-index   : 0.7129
CQR status        : ok
qhat              : 1.83 months
Coverage (events) : 0.8907
Mean width (all)  : 52.4 months

## Monte Carlo outputs
- mc_prob_survive_12_months
- mc_prob_survive_24_months
- mc_prob_survive_36_months
- mc_prob_survive_60_months
- mc_survival_p10_months / mc_survival_p50_months / mc_survival_p90_months
- mc_rmst_60_months

## Files
- main_predictions.csv
- monte_carlo_survival_test_predictions.csv
- tuned_model_predictions.csv
- tuned_model_coefficients.csv
- tuned_model_metrics.json
- metrics.json
- consistency_checks.json
- audit.json
- final_locked_model.pkl