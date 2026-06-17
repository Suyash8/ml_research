# Cox ENet + calibrated horizon risk estimation + Monte Carlo

## What this pipeline reports
- risk score from a fitted Cox Elastic-Net model
- horizon-wise calibrated event and survival probabilities at 12 / 24 / 36 / 60 months
- C-index on train / calibration / test
- horizon AUROC and Brier score on known-label test rows
- Monte Carlo survival summaries from the fitted Cox model and Breslow baseline hazard
- bounded RMST at 60 months

## Important interpretation note
The Monte Carlo block does not predict an exact survival lifespan.
It gives simulation-based summaries under the fitted Cox model and
the estimated Breslow baseline hazard.
The p90 simulated survival month is an upper-plausible survival time, not a guarantee.

## Files
- main_predictions.csv
- monte_carlo_survival_test_predictions.csv
- time_dependent_horizon_metrics.csv
- time_dependent_horizon_predictions.csv
- tuned_model_predictions.csv
- tuned_model_coefficients.csv
- tuned_model_metrics.json
- metrics.json
- consistency_checks.json
- audit.json
- final_locked_model.pkl