# Collinearity Threshold Sweep

Generated: 2026-04-19 16:04:03

## Run Comparison
| threshold | expr_after | expr_dropped | c_index_test | coverage_test_events_only | mean_interval_width_months_test | delta_test_c_index_vs_no_drop | delta_coverage_vs_no_drop | delta_width_vs_no_drop | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | 500 | 0 | 0.7240 | 0.8087 | 59.6737 | 0.0000 | 0.0000 | 0.0000 | skipped_existing |
| 0.9500 | 369 | 131 | 0.7209 | 0.8197 | 59.8411 | -0.0031 | 0.0109 | 0.1675 | skipped_existing |
| 0.9000 | 234 | 266 | 0.7193 | 0.8361 | 59.7008 | -0.0047 | 0.0273 | 0.0271 | skipped_existing |
| 0.8500 | 151 | 349 | 0.7136 | 0.8033 | 57.4582 | -0.0104 | -0.0055 | -2.2155 | skipped_existing |
| 0.7500 | 57 | 443 | 0.7203 | 0.8197 | 58.0784 | -0.0037 | 0.0109 | -1.5953 | skipped_existing |

## Key Selections
- Best test c-index: threshold=1.00, c-index=0.7240
- Tightest mean interval width: threshold=0.85, width=57.4582 months

## Notes
- Threshold 1.00 acts as no-drop baseline (since dropping is applied only when correlation > threshold).
- Expression drop counts are based on train split only and then applied consistently to calibration/test.