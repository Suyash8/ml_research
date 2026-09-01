# v3 — simplified rewrite

Written in response to faculty comments on v2. The draft was rebuilt from the
result files rather than edited, because simplifying complex prose sentence by
sentence produces simplified complex prose.

Status: compiles clean, 15 pages, all mechanical checks pass.

## What the comments asked for, and what was done

**Title too long, "Calibrated" and "from Multi-Omic TCGA Data" marked extra and
redundant.**
Now "Explainable Survival Prediction from Gene Expression and Clinical Data".
Nine words, no method name, no dataset name, no self-praise.

**Introduction should not have subsections.**
The `1.1 Motivation and Current Scenario` and `1.2 Key Contributions and
Architecture Overview` headings are gone. The introduction is five paragraphs:
motivation, background, existing work, the gap, contributions. Each paragraph
ends on something the next one picks up.

**"Where is architecture overview?? No need of subheadings."**
The v2 heading promised an architecture overview that the section did not
deliver. The heading is gone and the figure is introduced where it belongs, at
the end of the contributions paragraph.

**Sentences run for four or five lines.**
Longest sentence in the body is now 25 words. Mean is 12.8 across 392 sentences.
Verified by `scripts/check_paper.py`, not by eye.

**Language too complex for a biology or computer science PhD to follow.**
Rewritten for a first-year undergraduate. Jargon is defined in one short
sentence before it does any work: hazard, right censoring, concordance index,
Brier score, area under the curve, one-hot encoding, loading matrix.

**Full form at first use of every acronym, including common ones.**
Eleven acronyms are checked automatically: TNM, RNA-Seq, TCGA, PCA, SHAP, LIME,
L-BFGS-B, GBM, LIHC, PAAD, SKCM.

**"Full form is wrong" against LIHC, annotated HCC.**
LIHC is a TCGA dataset identifier, not a clinical abbreviation for hepatocellular
carcinoma. The text now says so explicitly, and Table 8 carries a separate TCGA
code column so the disease name and the dataset code are never conflated.

**Every claim needs a verified reference.**
47 references, each checked against the publisher record or PubMed. Five
carried-over entries were wrong and two were deleted. See
`../shared/REFERENCE_AUDIT.md`.

**No empty sentences.**
Sentences that only restated the previous sentence, announced what was coming,
or gestured at importance were cut. Body prose dropped from roughly 5,800 to
5,018 words while gaining content.

**Passive voice, no first person.**
Zero occurrences of we, our, us, I, my. Checked automatically.

## Two corrections that change the science

**The tumour mutational burden result was removed.**
v2 reported p = 0.028, q = 0.044, rank-biserial effect size 0.075 for tumour
mutational burden against the event indicator. No mutational burden column exists
in `data/analysis_reports/univariate_tests_vs_os_event.csv`, and no p-value near
0.028 appears in that file. The string "TMB" appears only in HTML diagrams under
`results/`, never in a statistical output. The claim could not be traced, so it
was deleted along with its citation.

**The collinearity filter was renamed to what it actually is.**
v1 and v2 called it an "incremental Gram-Schmidt collinearity filter (cosine
threshold 0.75)". The implementation is
`_drop_by_full_correlation` in `archive/cox_enet_calibrated_mc_pipeline_v5.py`,
and the summary file records `"method": "full_correlation_matrix"`. What the code
does: compute the absolute Pearson correlation matrix over the 500 candidate
columns on the training split, take the strict upper triangle, then discard any
column whose absolute correlation with an earlier column exceeds 0.75.

There is no Gram-Schmidt orthogonalisation, no cosine similarity, and no
incremental projection against a kept subspace. The three named properties were
all absent. v3 describes the greedy correlation screen it really is. The
numerical results are unchanged, because only the description was wrong.

Note that `scripts/plot_gram_schmidt_collinearity.py` and the figure axis labels
still carry the old name. The figure captions in v3 avoid it, but the plotting
script should be renamed and its titles corrected before submission.

## Limitations added

v2 listed three. v3 lists six. The three new ones:

- Both selected hyperparameters sit at the upper edge of their search grids.
  `alpha = 3.0` was the largest penalty strength searched and `l1_ratio = 0.7`
  the largest balance value. A stronger penalty was never tried. Confirmed
  against `hyperparameter_cv_results.csv`, a 25-point grid.
- Each patient has one sample taken at diagnosis, so progression over time is
  not modelled.
- The statistical screen is single-variable and shows association only.

The comparison against published work was also softened. v2 claimed the test
concordance index "aligns with published Cox Elastic-Net benchmarks on comparably
sized multi-omic TCGA cohorts" without naming one. v3 names the single
comparison it can support, Chaudhary et al. at 0.68, and states that the cohorts
and splits differ so it is a reference point rather than a controlled comparison.

## Number provenance

Every figure in the text was traced to a file before being written down.

| Claim | Source |
|---|---|
| N = 1620; splits 972 / 324 / 324; event rates 58.4 / 58.6 / 58.3 % | `results/cox_enet_calibrated_mc_outputs_v5/consistency_checks.json` |
| C-index 0.749 / 0.746 / 0.736; CV 0.722 ± 0.021 | `metrics.json`, `results/explainability/xai_metrics.json` |
| alpha 3.0, l1_ratio 0.7, 51 iterations | `metrics.json` |
| grid `{0.1,0.3,0.8,1.5,3.0}` x `{0.0,0.1,0.3,0.5,0.7}` | `hyperparameter_cv_results.csv`, 25 rows |
| 500 -> 59 genes, 441 dropped, 36,205 pairs above 0.75 | `collinearity_summary.json` |
| ACSM2A/ACSM2B r = 0.996; FGA/FGB/FGG r > 0.98 | `collinearity_summary.json`, top pairs preview |
| 6 -> 20 clinical columns, 59 -> 50 components, X = 70 | `audit.json`, `feature_checks` |
| AUC 0.759 / 0.860 / 0.847 / 0.847; Brier 0.178 / 0.152 / 0.150 / 0.132 | `time_dependent_horizon_metrics.csv` |
| known-outcome counts 289 / 247 / 228 / 218; death rates .284 / .543 / .662 / .784 | same file |
| 5,000 draws; mean interval 107.8 mo; median 103.0 mo | `metrics.json`, `monte_carlo` block |
| 461 distinct event times; longest follow-up 218.4 mo | `consistency_checks.json`, `monte_carlo_baseline_meta` |
| GBM +0.772, SKCM -0.322, PC01 +0.304, AGE +0.293 | `results/explainability/xai_report.md` |
| gene block sum abs coef 2.814 vs clinical 2.086 | `results/explainability/xai_group_summary.csv` |
| PC01 top ten genes, CD24 through CXCL5 | `xai_report.md`, back-projection section |
| high-risk patient eta +0.965, P10 3.3 / P50 13.6 / P90 38.1, RMST 16.9 | `main_predictions.csv`, patient TCGA-08-0348 |
| low-risk patient eta -1.578, P10 20.8 / P50 152.2, P90 at cap, RMST 52.3 | `main_predictions.csv`, patient TCGA-ZP-A9D0 |
| TCGA-DD-AACJ eta -0.028, 69.0 mo censored, 70 terms sum exactly | `xai_patient_feature_contributions_test.csv`, sum -0.028471 |
| cohort sizes and median survival 12.16 / 15.34 / 19.76 / 35.78 | `data/analysis_reports/outcome_by_cancer_type.csv` |
| 1,049 columns; cancer type q = 1.36e-51; stage q = 5.15e-43 | `data/analysis_reports/statistical_analysis_report.md` |
| DKFZp547D155 rho -0.346; CA9 -0.338; CXCL5 -0.308 | same file, Spearman section |

One narrative detail was corrected while checking. v2 said the largest
risk-increasing terms for TCGA-DD-AACJ were age and one genomic component, with
brain cancer non-membership among the protective terms. The largest single term
is in fact the protective one, brain cancer non-membership at -0.616, ahead of
age at +0.328. v3 states the ordering as it appears in the file.

## Still needs a human

- Author names and affiliations are placeholders.
- Acknowledgments and the generative AI declaration are placeholders.
- The data availability statement should name the exact cBioPortal study
  identifiers and the download date.
- `scripts/plot_gram_schmidt_collinearity.py` and the two collinearity figures
  still carry "Gram-Schmidt" in their titles and axis labels. The figures need
  regenerating with accurate labels.

  **Confirmed by visual inspection of page 5.** Both colorbars in Figure 2 read
  "Cosine Similarity |CosSim|", and panel (b) is annotated "(|r| <= 0.75)". The
  caption and Section 3.2 correctly describe an absolute Pearson correlation. The
  figure therefore contradicts its own caption on the same page, and it uses two
  different names for the same quantity within one figure. Nothing in the LaTeX
  source is wrong; the defect exists only in the rendered PNG files. Regenerate
  both panels with the colorbar labelled "Absolute Pearson correlation |r|"
  before submission.
- The proportional hazards assumption is still untested. Schoenfeld residuals are
  cheap to compute and would close the largest methodological gap.
