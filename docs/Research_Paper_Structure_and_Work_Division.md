# Multi-Omic Cox Elastic-Net Survival Analysis (v5)
## Research Paper Blueprint, Structural Specification & Team Task Division Plan

---

## 📌 Executive Summary & Team Context

* **Project:** Multi-Omic Cox Elastic-Net Pipeline (v5) for Cancer Survival Analysis & Patient-Specific XAI.
* **Dataset:** TCGA Pan-Cancer (GBM, LIHC, PAAD, SKCM — $N=1,628$ patients, $1,049$ initial features).
* **Team Roster:**
  * **Suyash & Addy:** Developers & Lead Researchers who authored the codebase, model architecture, dynamic programming loss, calibration, Monte Carlo engine, and PCA back-projection math.
  * **Sowhardya & Sriparna:** Co-Authors handling literature synthesis, background context, results visualization, formatting, and conclusions.
  * *Note:* The background historical subsection (assigned to Sray) is handled separately outside the 4 team members.

---

## 📖 SECTION 1: Paper Structure & Detailed Blueprint

```
+-----------------------------------------------------------------------------------+
|                            RESEARCH PAPER BLUEPRINT                                |
+-----------------------------------------------------------------------------------+
| Section 1: Introduction & Literature Survey      | ~1.75 Pages (3 Columns/Layout)  |
| Section 2: Proposed Method (Math & Derivations)   | ~2.00 Pages                    |
| Section 3: Results (Block-by-Block Metrics)       | ~2.00 Pages                    |
| Section 4: Discussion (Inference & Validation)    | ~1.00 Page                     |
| Section 5: Conclusion                             | ~0.50 Page (1 Column Format)   |
+-----------------------------------------------------------------------------------+
```

---

### Section 1: Introduction + Literature Survey (~1.75 Pages)
* **1.1 Motivation & Clinical Context:**
  * Cancer heterogeneity renders single-modality clinical staging (TNM) insufficient for fine-grained survival prognosis.
  * High-dimensional transcriptomics (RNA-Seq) contains critical biological signals but suffers from the $P \gg N$ curse of dimensionality and severe multicollinearity.
* **1.2 Current Scenario in Survival Analysis:**
  * Standard Cox Proportional Hazards models fail when $P > N$.
  * Deep learning models (e.g., CoxPASNet, DeepSurv) act as black boxes, lack clinical interpretability, and require massive sample sizes.
  * Time-varying covariate approaches fail on cross-sectional genomic datasets like TCGA due to biopsy sparsity.
* **1.3 Biological & Machine Learning Literature Survey:**
  * *Biological Perspective:* Oncogenic pathway activation, dysregulated transcripts (e.g., *CST1*, *CXCL5*, *ETNPPL*), and tumor mutational burden (TMB) as drivers of patient mortality.
  * *Machine Learning Perspective:* Regularization techniques (Ridge, LASSO, Elastic-Net), dimensionality reduction (PCA, UMAP), non-parametric calibration (PAVA), and survival time simulation.
* **1.4 Identified Research Gaps:**
  1. *Gap 1 (Dimensionality & Collinearity):* Existing multi-omic pipelines retain collinear genes, causing coefficient variance explosion.
  2. *Gap 2 (Uncalibrated Hazard Ratios):* Standard Cox models output relative risk scores ($\eta$), which clinicians cannot convert into absolute 1-to-5 year survival probabilities.
  3. *Gap 3 (Lack of Uncertainty Bounds):* Point estimates fail to quantify individual patient variance.
  4. *Gap 4 (PCA Black-Box Barrier):* PCA reduces dimensionality but hides gene-level biological mechanisms.
* **1.5 How Our Work Solves These Gaps (Key Contributions):**
  * Gram-Schmidt collinearity filtering ($r > 0.75$) + 50 PCA latent component projection.
  * Custom Cox Elastic-Net with $O(N)$ dynamic programming Breslow partial log-likelihood.
  * Isotonic Calibration (PAVA) for absolute survival probability mapping at 12, 24, 36, and 60 months.
  * 5,000-draw Monte Carlo inverse transform sampling for $P10$, $P50$ (median), $P90$, and RMST uncertainty bounds.
  * Novel closed-form PCA Back-Projection ($W_{\text{gene}} = V \cdot \beta_{\text{pca}}$) yielding exact patient-level additive risk waterfalls ($\Delta \eta_{g,i}$).
* **1.6 Architecture Block Diagram:**
  * High-resolution diagram showing end-to-end data flow from raw TCGA files to calibrated survival dossiers.

---

### Section 2: Proposed Method (~2.00 Pages)
*Connected, running mathematical derivations (no isolated floating equations):*

1. **Feature Preprocessing & Gram-Schmidt Collinearity Filtering:**
   $$\text{Cosine Similarity}(g_a, g_b) = \frac{\langle g_a, g_b \rangle}{\|g_a\|_2 \|g_b\|_2} > 0.75 \implies \text{Drop } g_b$$
2. **Latent PCA Projection:**
   $$X_{\text{pca}} = Z_{\text{gene}} \cdot V \in \mathbb{R}^{N \times 50}, \quad V \in \mathbb{R}^{300 \times 50}$$
3. **Unified Feature Matrix:**
   $$X = [X_{\text{clin}} \mid X_{\text{pca}}] \in \mathbb{R}^{N \times 59}$$
4. **Cox Elastic-Net Objective with $O(N)$ Dynamic Programming:**
   * Log-partial likelihood over event times $t \in D$:
     $$\ell(\beta) = \sum_{i \in D} \left( \eta_i - \ln \sum_{j \in R(t_i)} \exp(\eta_j) \right)$$
   * Dynamic programming suffix sum optimization via `np.cumsum`:
     $$S^{(0)}(t) = \sum_{j \in R(t)} \exp(\eta_j), \quad S^{(1)}(t) = \sum_{j \in R(t)} \exp(\eta_j) X_j$$
   * Smooth Elastic-Net Loss:
     $$\mathcal{L}(\beta) = -\ell(\beta) + \alpha \left[ \rho \sum_{k} \sqrt{\beta_k^2 + \epsilon} + \frac{1-\rho}{2} \|\beta\|_2^2 \right]$$
5. **Isotonic Calibration Layer (PAVA):**
   $$P(S > t \mid \eta_i) = 1 - f_{\text{iso}, t}(\eta_i)$$
6. **Monte Carlo Inverse Transform Survival Simulation:**
   $$\Lambda_{\text{target}}^{(k)} = \frac{-\ln(U^{(k)})}{\exp(\eta_i)}, \quad U^{(k)} \sim \text{Uniform}(0,1)$$
   $$t^{(k)} = \Lambda_0^{-1}\left( \Lambda_{\text{target}}^{(k)} \right) \implies \text{Extract } P10, P50, P90, \text{RMST}$$
7. **PCA Back-Projection XAI Unrolling:**
   $$\eta_{\text{pca}} = X_{\text{pca}} \cdot \beta_{\text{pca}} = (Z_{\text{gene}} \cdot V) \cdot \beta_{\text{pca}} = Z_{\text{gene}} \cdot (V \cdot \beta_{\text{pca}})$$
   $$W_{\text{gene}} = V_{300 \times 50} \cdot \beta_{\text{pca}} \in \mathbb{R}^{300}$$
   $$\Delta \eta_{g, i} = Z_{g, i} \cdot W_{\text{gene}, g}$$

---

### Section 3: Results (Block-by-Block Performance) (~2.00 Pages)
* **Block 1: Cohort Ingestion & Feature Reduction Results:**
  * Table showing dimensionality changes ($1,049 \to 300 \text{ genes} \to 50 \text{ PCs} + 9 \text{ clinical} = 59 \text{ total}$).
* **Block 2: Cox Elastic-Net Model Performance:**
  * Harrell's Concordance Index (C-Index) across Train, Validation, and Test splits.
  * Comparison against baseline models (Standard Cox, Ridge-only, LASSO-only).
* **Block 3: Isotonic Calibration Metrics:**
  * Calibration curves (Observed vs Predicted survival probabilities at 12, 24, 36, 60 months).
  * Brier score improvement post-calibration.
* **Block 4: Monte Carlo Simulation Quantiles:**
  * Patient trajectory distribution plots ($P10, P50, P90$).
  * RMST evaluation at 60 months.
* **Block 5: Patient-Specific XAI Waterfall Results:**
  * Global gene ranking table (Top 10 Risk vs Protective genes).
  * Individual patient case studies (`TCGA-DD-AAEE` low risk vs `TCGA-BF-A3DL` high risk).

---

### Section 4: Discussion (~1.00 Page)
* **4.1 Why This Pipeline Outperforms Existing Approaches:**
  * Solves $P \gg N$ instability while maintaining exact gene-level interpretability.
  * Eliminates black-box approximations (SHAP/LIME) in favor of closed-form matrix unrolling.
* **4.2 Clinical Actionability & Inference:**
  * Translates abstract log-hazards ($\eta$) into real-world survival windows ($P10 - P90$).
* **4.3 Monte Carlo Validation & Robustness:**
  * Rationale behind 5,000 stochastic draws over the Breslow baseline hazard.

---

### Section 5: Conclusion (~0.50 Page — 1 Column Format)
* Summary of the multi-omic framework, mathematical contributions, clinical utility, and future extension to single-cell multi-omics.

---

## 👥 SECTION 2: Work Division Matrix by Name

```
+-----------------------------------------------------------------------------------+
|                              WORK DIVISION MATRIX                                 |
+---------------+-------------------------------------+-----------------------------+
| Name          | Primary Section Ownership           | Key Deliverables            |
+---------------+-------------------------------------+-----------------------------+
| Suyash        | Section 2: Proposed Method          | Mathematical Derivations,   |
|               | (Algorithmic & Formulative Engine)  | Dynamic Programming Loss,   |
|               |                                     | PCA Back-Projection Math    |
+---------------+-------------------------------------+-----------------------------+
| Addy          | Section 3: Results & Experiments    | Generating Tables/Plots,    |
|               | Section 4.1-4.2: Discussion         | C-Index Metrics, Calibration|
|               |                                     | & Monte Carlo Results       |
+---------------+-------------------------------------+-----------------------------+
| Sowhardya     | Section 1: Introduction & Gaps      | Clinical Motivation, Gaps,  |
|               | Literature Survey (Bio & ML)        | Literature Tables, Diagram  |
|               |                                     | Rendering                   |
+---------------+-------------------------------------+-----------------------------+
| Sriparna      | Section 4.3: Discussion (Validation)| Discussion Synthesis,       |
|               | Section 5: Conclusion (1-Col)       | Formatting, References,     |
|               | Master Editing & Proofreading       | Single-Column Formatting    |
+---------------+-------------------------------------+-----------------------------+
```

---

### 📋 Detailed Task Breakdown per Team Member

#### 👤 Suyash
* **Primary Ownership:** **Section 2: Proposed Method** (~2.0 pages)
* **Detailed Tasks:**
  1. Write the running mathematical formulation connecting all equations linearly without floating disconnected boxes.
  2. Write Section 2.1: Gram-Schmidt Cosine Collinearity Filter equation and algorithm.
  3. Write Section 2.2: Latent PCA Transformation $X_{\text{pca}} = Z_{\text{gene}} \cdot V$.
  4. Write Section 2.3: Custom Cox Elastic-Net Partial Log-Likelihood $\ell(\beta)$ and the $O(N)$ dynamic programming suffix sum implementation (`np.cumsum`).
  5. Write Section 2.4: Isotonic Calibration PAVA step-mapping formulation $P(S > t) = 1 - f_{\text{iso}, t}(\eta)$.
  6. Write Section 2.5: Monte Carlo Inverse Transform Sampling formulas $\Lambda_{\text{target}}^{(k)} = \frac{-\ln(U^{(k)})}{\exp(\eta_i)}$ and baseline inversion.
  7. Write Section 2.6: PCA Back-Projection XAI derivation $W_{\text{gene}} = V \cdot \beta_{\text{pca}}$ and patient waterfall equation $\Delta \eta_{g,i} = Z_{g,i} \cdot W_{\text{gene}, g}$.

---

#### 👤 Addy
* **Primary Ownership:** **Section 3: Results** (~2.0 pages) & **Section 4.1-4.2: Discussion** (~0.5 pages)
* **Detailed Tasks:**
  1. Extract and format empirical performance metrics from the codebase (`C-Index`, Brier Scores, RMST).
  2. Write Section 3.1: Feature Reduction and Cohort Processing Table ($1,049 \to 300 \to 50 + 9 = 59$).
  3. Write Section 3.2: Model Performance Table comparing Cox Elastic-Net against baseline standard Cox and LASSO models.
  4. Write Section 3.3: Isotonic Calibration Results (Observed vs Calibrated probabilities at 12m, 24m, 36m, 60m).
  5. Write Section 3.4: Monte Carlo Simulation Results ($P10, P50, P90$ quantile distributions).
  6. Write Section 3.5: XAI Waterfall Results (Global gene weight table + Patient case studies for `TCGA-DD-AAEE` and `TCGA-BF-A3DL`).
  7. Write Section 4.1: Technical discussion on why our pipeline outperforms black-box models.

---

#### 👤 Sowhardya
* **Primary Ownership:** **Section 1: Introduction & Literature Survey** (~1.75 pages)
* **Detailed Tasks:**
  1. Write Section 1.1: Clinical motivation on multi-omic cancer prognosis and TNM staging limitations.
  2. Write Section 1.2: Current scenario in survival analysis (Cox model failures in $P \gg N$, deep learning black-box limitations).
  3. Write Section 1.3: Dual Literature Survey:
     * *Biological Perspective:* Role of dysregulated transcripts, oncogenes, and mutation burden.
     * *Machine Learning Perspective:* Regularization, PCA, PAVA calibration, and survival simulation.
  4. Write Section 1.4: Research Gaps identification (Collinearity, Uncalibrated risk, Uncertainty bounds, PCA interpretability barrier).
  5. Write Section 1.5: How our proposed work bridges these gaps.
  6. Insert and caption the Architecture Diagram (`component_io_block_diagram.html` / `pipeline_block_diagram.html`).

---

#### 👤 Sriparna
* **Primary Ownership:** **Section 4.3: Discussion**, **Section 5: Conclusion**, & **Master Formatting** (~1.25 pages)
* **Detailed Tasks:**
  1. Write Section 4.3: Monte Carlo simulation validation, bootstrap interpretation, and clinical utility.
  2. Write Section 5: Conclusion (Single-column layout as requested) summarizing key findings and future directions.
  3. Compile all team members' sections into a single cohesive manuscript (LaTeX or Word template).
  4. Ensure strict page budget adherence (Section 1: ~1.75 pages, Section 2: ~2.0 pages, Section 3: ~2.0 pages, Section 4: ~1.0 page, Section 5: ~0.5 page).
  5. Verify reference formatting, equation numbering, table styling, and figure captions across the document.

---

## 🎯 Verification & Next Steps

1. This updated blueprint and work division matrix has been saved to:
   [docs/Research_Paper_Structure_and_Work_Division.md](file:///home/illionar/Projects/ml_research/docs/Research_Paper_Structure_and_Work_Division.md)
2. Each team member (Suyash, Addy, Sowhardya, Sriparna) can take their assigned section directly from this document!
