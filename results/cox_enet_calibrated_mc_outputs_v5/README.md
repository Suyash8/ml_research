# Cox ENet + calibrated horizon risk estimation + Monte Carlo

## What this pipeline reports
- risk score from a fitted Cox Elastic-Net model
- horizon-wise calibrated event and survival probabilities at 12 / 24 / 36 / 60 months
- C-index on train / calibration / test
- horizon AUROC and Brier score on known-label test rows
- Monte Carlo survival summaries from the fitted Cox model and Breslow baseline hazard
- bounded RMST at 60 months