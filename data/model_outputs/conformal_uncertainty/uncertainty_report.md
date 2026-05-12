# Survival Conformal Uncertainty Report

## Conformal Method
CQR (Romano, Patterson & Candès, NeurIPS 2019) on the **raw month scale**.
No log transform — intervals are symmetric in months, not exponentially skewed.

- alpha             : 0.1 (target coverage 90%)
- qhat              : 1.22 months
- status            : ok
- coverage (events) : 0.8677248677248677

## Interval Width Summary
| Group | Mean width (months) | Median width (months) |
|---|---|---|
| All test patients | 65.5 | 56.7 |
| High-uncertainty | 76.4 | 73.7 |
| Low-uncertainty  | 44.7 | 39.2 |

## Uncertainty Flags
Total test patients: **324**

| Type | N | % | Description |
|---|---|---|---|
| type1_model_uncertain | 81 | 25.0% | Raw QR interval width > Q75 — model has high aleatoric uncertainty |
| type2_risk_ambiguous | 163 | 50.3% | Cox risk score in IQR [Q25, Q75] — borderline discriminative zone |
| type3_upper_tail | 81 | 25.0% | Conformal hi > Q75 of test hi — long predicted survival, poorly bounded |
| high_uncertainty | 212 | 65.4% | Triggered >= 1 uncertainty type |
| very_high_uncertainty | 82 | 25.3% | Triggered >= 2 uncertainty types — highest clinical attention needed |

## Uncertainty Thresholds
- Uncertainty quantile: 0.75 (top 25% flagged per axis)
- Type 1 threshold (QR raw width): > 78.9 months
- Type 2 threshold (risk IQR): [-0.784, 0.870]
- Type 3 threshold (conformal hi): > 85.1 months

## Type Co-occurrence
| Pattern | N |
|---|---|
| Type 1 only (model_uncertain) | 1 |
| Type 2 only (risk_ambiguous)  | 129 |
| Type 3 only (upper_tail)      | 0 |
| Type 1 + 2                    | 32 |
| Type 1 + 3                    | 79 |
| Type 2 + 3                    | 33 |
| All three                     | 31 |
| None (low uncertainty)        | 112 |

## Top 20 Most Uncertain Patients
| PATIENT_ID   |   OS_MONTHS |   OS_EVENT |   risk_score |   conformal_lo |   conformal_hi |   conformal_width |   n_uncertainty_types | uncertainty_label                           |
|:-------------|------------:|-----------:|-------------:|---------------:|---------------:|------------------:|----------------------:|:--------------------------------------------|
| TCGA-EE-A29E |      63.73  |          0 |   -0.310052  |        4.78771 |       155.543  |          150.756  |                     3 | model_uncertain, risk_ambiguous, upper_tail |
| TCGA-ER-A2NH |      41.52  |          0 |   -0.703623  |        4.80268 |       133.972  |          129.169  |                     3 | model_uncertain, risk_ambiguous, upper_tail |
| TCGA-GN-A26D |      47.96  |          1 |   -0.696639  |        4.9137  |       124.227  |          119.313  |                     3 | model_uncertain, risk_ambiguous, upper_tail |
| TCGA-EE-A20F |      91.49  |          0 |   -0.771054  |        5.11177 |       122.487  |          117.376  |                     3 | model_uncertain, risk_ambiguous, upper_tail |
| TCGA-EE-A3AE |      54.47  |          0 |   -0.437335  |        5.5734  |       120.565  |          114.992  |                     3 | model_uncertain, risk_ambiguous, upper_tail |
| TCGA-ER-A199 |       9.17  |          1 |    0.302217  |        4.24433 |       117.331  |          113.086  |                     3 | model_uncertain, risk_ambiguous, upper_tail |
| TCGA-BF-A5EO |      23.09  |          0 |   -0.417718  |        2.07752 |       112.833  |          110.756  |                     3 | model_uncertain, risk_ambiguous, upper_tail |
| TCGA-BF-A3DL |      25.26  |          0 |    0.372671  |        4.20638 |       114.822  |          110.615  |                     3 | model_uncertain, risk_ambiguous, upper_tail |
| TCGA-W3-A828 |     120.99  |          1 |   -0.0254185 |        3.83402 |       107.643  |          103.809  |                     3 | model_uncertain, risk_ambiguous, upper_tail |
| TCGA-DA-A1HY |     144.78  |          0 |   -0.703153  |        4.26205 |       107.459  |          103.197  |                     3 | model_uncertain, risk_ambiguous, upper_tail |
| TCGA-XV-AAZV |      13.53  |          0 |   -0.782727  |        4.29956 |       105.343  |          101.043  |                     3 | model_uncertain, risk_ambiguous, upper_tail |
| TCGA-IH-A3EA |      17.21  |          0 |    0.12257   |        5.43885 |       104.052  |           98.6131 |                     3 | model_uncertain, risk_ambiguous, upper_tail |
| TCGA-D9-A1JX |       7.1   |          1 |    0.453907  |        4.20967 |       102.491  |           98.2809 |                     3 | model_uncertain, risk_ambiguous, upper_tail |
| TCGA-D3-A51G |      17.165 |          0 |   -0.435932  |        4.5691  |       101.89   |           97.3206 |                     3 | model_uncertain, risk_ambiguous, upper_tail |
| TCGA-WE-A8ZQ |      63.17  |          0 |   -0.139347  |        6.45217 |       103.577  |           97.1249 |                     3 | model_uncertain, risk_ambiguous, upper_tail |
| TCGA-ER-A42H |      13.99  |          1 |   -0.0705357 |        6.05594 |       102.728  |           96.6721 |                     3 | model_uncertain, risk_ambiguous, upper_tail |
| TCGA-FR-A8YD |      36.24  |          1 |   -0.290063  |        5.60078 |       102.13   |           96.5297 |                     3 | model_uncertain, risk_ambiguous, upper_tail |
| TCGA-ZP-A9D2 |      25.13  |          1 |   -0.476003  |        4.11368 |        99.5016 |           95.3879 |                     3 | model_uncertain, risk_ambiguous, upper_tail |
| TCGA-D3-A8GR |     129.53  |          1 |   -0.582473  |        4.92023 |        99.8331 |           94.9129 |                     3 | model_uncertain, risk_ambiguous, upper_tail |
| TCGA-XV-AAZW |      12.91  |          1 |   -0.0981032 |        4.64647 |        98.0184 |           93.372  |                     3 | model_uncertain, risk_ambiguous, upper_tail |

## Output Files
- `conformal_uncertainty_predictions.csv` — all test patients with uncertainty columns
- `high_uncertainty_patients.csv`          — only high_uncertainty patients, sorted by rank
- `very_high_uncertainty_patients.csv`     — very_high_uncertainty patients only
- `uncertainty_summary.json`               — full numeric summary
- `uncertainty_report.md`                  — this file