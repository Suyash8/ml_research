# v5 Explainability Report

This report is generated from the locked v5 Cox Elastic-Net artifact.
It explains the transformed feature space, not raw gene effects directly.

## Model Context
- train C-index: 0.7489
- calibration C-index: 0.7461
- test C-index: 0.7361

## Global Drivers
- cat__CANCER_TYPE_GBM: coef=0.7720 (risk_increasing)
- cat__CANCER_TYPE_SKCM: coef=-0.3215 (risk_decreasing)
- EXPR_PC01: coef=0.3044 (risk_increasing)
- num__AGE: coef=0.2934 (risk_increasing)
- EXPR_PC02: coef=-0.2076 (risk_decreasing)
- EXPR_PC04: coef=-0.1714 (risk_decreasing)
- EXPR_PC08: coef=0.1524 (risk_increasing)
- cat__AGE_GROUP_>75: coef=0.1495 (risk_increasing)
- cat__CANCER_TYPE_PAAD: coef=-0.1203 (risk_decreasing)
- EXPR_PC33: coef=0.1193 (risk_increasing)
- cat__RACE_Unknown: coef=0.1154 (risk_increasing)
- EXPR_PC38: coef=-0.1113 (risk_decreasing)

## Group Summary
- expression: n=50, sum_abs_coef=2.8140
- clinical: n=20, sum_abs_coef=2.0863

## Expression Back-Projection
- EXPR_PC01 -> EXPR_CD24: loading=0.1786, weighted=0.0544
- EXPR_PC01 -> EXPR_CXCL14: loading=0.1734, weighted=0.0528
- EXPR_PC01 -> EXPR_GPRC5A: loading=0.1672, weighted=0.0509
- EXPR_PC01 -> EXPR_PPP1R1B: loading=0.1666, weighted=0.0507
- EXPR_PC01 -> EXPR_SYT13: loading=0.1664, weighted=0.0506
- EXPR_PC01 -> EXPR_SERPINA3: loading=0.1621, weighted=0.0493
- EXPR_PC01 -> EXPR_CA9: loading=0.1620, weighted=0.0493
- EXPR_PC01 -> EXPR_ADAM6: loading=0.1600, weighted=0.0487
- EXPR_PC01 -> EXPR_FGFR2: loading=0.1584, weighted=0.0482
- EXPR_PC01 -> EXPR_CXCL5: loading=0.1583, weighted=0.0482
- EXPR_PC02 -> EXPR_HEPACAM: loading=0.2555, weighted=-0.0530
- EXPR_PC02 -> EXPR_AQP4: loading=0.2483, weighted=-0.0515
- EXPR_PC02 -> EXPR_ABCB5: loading=-0.2258, weighted=0.0469
- EXPR_PC02 -> EXPR_ESRP1: loading=-0.2248, weighted=0.0467
- EXPR_PC02 -> EXPR_TCN1: loading=-0.2182, weighted=0.0453
- EXPR_PC02 -> EXPR_FDCSP: loading=-0.2127, weighted=0.0442
- EXPR_PC02 -> EXPR_NEU4: loading=0.2053, weighted=-0.0426
- EXPR_PC02 -> EXPR_EDM1: loading=-0.2051, weighted=0.0426
- EXPR_PC02 -> EXPR_CTNND2: loading=0.2009, weighted=-0.0417
- EXPR_PC02 -> EXPR_CACNG4: loading=0.1951, weighted=-0.0405

## Interpretation Notes
- Clinical features are directly interpretable in transformed form.
- Expression coefficients live on PCA components; gene-level meaning comes from back-projection.
- Patient-level contributions are additive on the Cox log-risk scale.
- The Monte Carlo survival outputs remain uncertainty summaries, not exact lifespan predictions.