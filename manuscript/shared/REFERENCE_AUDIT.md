# Reference audit

Every entry in `references.bib` was checked against the publisher record or
PubMed before v3 was written. Author lists were checked name by name, because
that is the field where a wrong citation looks right.

Five entries carried over from v1 and v2 were wrong. Two of them had invented
co-authors. One did not appear to exist at all.

## Corrected

**`vanhouwelingen2006crossvalidated`** — fabricated author list.

The v1/v2 entry listed `Simon, Ivo`, `van Tol, Jan H.` and `Vermeulen, Lodewijk`.
None of the three are on the paper. Verified against PubMed 16143967, the real
authors are van Houwelingen HC, Bruinsma T, Hart AA, van't Veer LJ, Wessels LF.
Title, journal, year, volume and pages were all correct, which is why the error
survived two drafts.

**`lee2019dynamic` -> `lee2020dynamic`** — fabricated co-author, wrong year.

The v1/v2 entry listed `Zame, William R.` as second author. Zame is an author on
DeepHit but not on Dynamic-DeepHit. Verified against PubMed 30951460, the authors
are Lee C, Yoon J, van der Schaar M. The print year is 2020, volume 67, issue 1,
pages 122-133.

**`hao2018coxpasnet` -> `hao2019coxpasnet`** — wrong year.

The DOI in the entry, `10.1186/s12920-019-0624-2`, resolves to a 2019 article in
BMC Medical Genomics. The entry was dated 2018. Key renamed to match.

**`giatromanolaki2001ca9` -> `giatromanolaki2001carbonic`** — wrong title, wrong issue.

The v1/v2 entry gave the title as "Carbonic Anhydrase 9 Relates to Hypoxia
Inducible Factor and Vascular Endothelial Growth Factor Expression in Non-small
Cell Lung Cancer" and the issue as 19. The actual article is "Expression of
hypoxia-inducible carbonic anhydrase-9 relates to angiogenic pathways and
independently to poor outcome in non-small cell lung cancer", Cancer Research
2001;61(21):7992-7998.

**`grunkin2011cxcl5` -> `zhou2012overexpression`** — key contradicted the entry.

The citation key claimed a 2011 paper by Grunkin. The entry body correctly
described Zhou et al., Hepatology 2012;56(6):2242-2254. The key was renamed so
that an author-year citation style cannot render the wrong name.

**`poirion2021deepprog`** — wrong given name.

Second author was listed as `Jing, Zixiao`. The published author list reads
Poirion OB, Jing Z (Zheng Jing), Chaudhary K, Huang S, Garmire LX.

## Deleted

**`jain2024isotonic`** — could not be verified to exist.

The entry read `Jain, Anshul and others`, "Isotonic Survival Regression:
Calibrated Survival Distributions from Deep Cox Models", arXiv preprint, 2024,
with no arXiv identifier and no DOI. Searches returned no such paper. The nearest
real work is on calibrated survival distributions and isotonic distributional
regression by other authors.

The v2 draft used it to support the sentence introducing isotonic calibration.
That sentence is now supported by the primary sources instead:
`ayer1955empirical` for the pool adjacent violators algorithm, and
`zadrozny2002transforming` with `niculescu2005predicting` for its use as a
probability calibrator.

**`chalmers2017mutational`** — the claim it supported was itself unsupported.

The v2 draft reported that tumour mutational burden reached nominal significance
in this cohort, at p = 0.028, q = 0.044, rank-biserial effect size 0.075. No
mutational burden column exists in
`data/analysis_reports/univariate_tests_vs_os_event.csv`, and no value near
p = 0.028 appears anywhere in that file. The string "TMB" occurs only in the
HTML diagrams under `results/`, never in a statistical output. The claim was
removed, and this reference went with it rather than being repurposed.

## Added

The core method of the paper was never cited in v1 or v2. Cox 1972 is now cited,
along with the other primary sources the method actually rests on.

| Key | Why it was needed |
|---|---|
| `cox1972regression` | the proportional hazards model itself |
| `kaplan1958nonparametric` | the survival estimator used as the calibration target |
| `zou2005regularization` | the elastic-net penalty |
| `hotelling1933analysis`, `jolliffe2016principal` | principal component analysis |
| `byrd1995limited` | the L-BFGS-B optimiser |
| `ayer1955empirical`, `niculescu2005predicting` | isotonic regression and probability calibration |
| `graf1999assessment` | the Brier score for censored data |
| `heagerty2005survival` | time-dependent area under the curve |
| `royston2013restricted` | restricted mean survival time |
| `uno2011c` | censoring-adjusted concordance |
| `mann1947test`, `spearman1904proof`, `benjamini1995controlling` | the three tests used in the statistical screen |
| `amin2017ajcc` | the TNM staging system |
| `cerami2012cbio`, `gao2013integrative` | cBioPortal, the actual download source |
| `brennan2013somatic`, `tcga2017hepatocellular`, `tcga2017pancreatic`, `tcga2015melanoma` | the four cohorts, individually |
| `li2011rsem`, `love2014moderated` | RNA-Seq quantification and the count distribution |
| `lundberg2017unified`, `ribeiro2016why` | SHAP and LIME, named in the text and previously uncited |
| `suissa2008immortal` | immortal time bias, previously asserted without a source |
| `pedregosa2011scikit` | the library used |
| `hastie2009elements` | textbook treatment of the penalties |

## Claims re-attributed

The v2 draft opened with a claim that staging captures anatomy rather than
transcriptional programmes, cited to `vanhouwelingen2006crossvalidated`. That is
a methods paper on cross-validated Cox regression and does not support a claim
about staging. The sentence now cites `amin2017ajcc`, the staging manual.

## Statistical language corrected

Two phrases in v2 misdescribed the tests that were run.

Tumour stage was called "independently significant". The analysis is a
single-variable screen, which cannot establish independence from other
variables. The text now states plainly that the screen shows association and not
independent effect.

Genes were said to "pass genome-wide significance". That term refers to the
genome-wide threshold used in association studies, not to a small corrected
p-value from a 1,049-column screen. The phrase was removed.
