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
  * **Figure 1 File:** `results/plots/fig1_pipeline_block_diagram.png`

---

### Section 2: Proposed Method (~2.00 Pages)
*Fully Formatted Running Mathematical Derivations & Closed-Form Formulations:*

#### 2.1 Feature Preprocessing & Gram-Schmidt Collinearity Filter
Let $G \in \mathbb{R}^{N \times P_{\text{raw}}}$ represent the raw transcriptomic gene expression matrix across $N$ patients and $P_{\text{raw}} = 300$ high-variance genes. To eliminate linear redundancy and coefficient variance explosion, we apply a Gram-Schmidt orthogonalization-inspired cosine collinearity filter:

$$\text{CosSim}(g_a, g_b) = \frac{\langle g_a, g_b \rangle}{\|g_a\|_2 \|g_b\|_2} = \frac{\sum_{i=1}^{N} g_{i,a} g_{i,b}}{\sqrt{\sum_{i=1}^{N} g_{i,a}^2} \sqrt{\sum_{i=1}^{N} g_{i,b}^2}}$$

$$\text{If } |\text{CosSim}(g_a, g_b)| > 0.75 \implies \text{Remove gene } g_b$$

* **Figure 3 File:** `results/plots/fig3_gram_schmidt_collinearity.png`

#### 2.2 Standardized Latent Principal Component Projection
Following collinearity pruning, the expression matrix is log-transformed $Z_{\text{gene}} = \text{StandardScaler}(\log_2(G + 1))$ and projected onto $K = 50$ orthogonal principal components via Singular Value Decomposition (SVD):

$$Z_{\text{gene}} = U \Sigma V^T \implies X_{\text{pca}} = Z_{\text{gene}} V \in \mathbb{R}^{N \times 50}$$

Where $V \in \mathbb{R}^{300 \times 50}$ is the right-singular eigenvector loading matrix satisfying $V^T V = I_{50}$.

#### 2.3 Unified Model Input Feature Matrix
The clinical features $X_{\text{clin}} \in \mathbb{R}^{N \times 9}$ (Age, Stage, Sex, TMB) are standardized and concatenated with the latent genomic matrix $X_{\text{pca}}$ to form the final model input matrix $X$:

$$X = \left[ X_{\text{clin}} \;\middle\|\; X_{\text{pca}} \right] \in \mathbb{R}^{N \times 59}$$

* **Figure 2 File:** `results/plots/fig2_component_io_block_diagram.png`

#### 2.4 Cox Elastic-Net Partial Log-Likelihood & $O(N)$ Dynamic Programming Suffix Sums
The hazard function for patient $i$ at time $t$ is formulated as:

$$h(t \mid X_i) = h_0(t) \exp(\eta_i), \quad \text{where } \eta_i = X_i \beta = \sum_{j=1}^{59} X_{i,j} \beta_j$$

For ordered event times $t_1 < t_2 < \dots < t_D$, the Breslow Cox log-partial likelihood is:

$$\ell(\beta) = \sum_{i \in D} \left[ \eta_i - \ln \left( \sum_{j \in R(t_i)} \exp(\eta_j) \right) \right]$$

To reduce computation from $O(N^2)$ to $O(N)$, we compute risk-set denominator suffix sums dynamically:

$$S^{(0)}(t_i) = \sum_{j \in R(t_i)} \exp(\eta_j), \quad S^{(1)}(t_i) = \sum_{j \in R(t_i)} \exp(\eta_j) X_j$$

$$\nabla \ell(\beta) = \sum_{i \in D} \left[ X_i - \frac{S^{(1)}(t_i)}{S^{(0)}(t_i)} \right]$$

#### 2.5 Smooth Elastic-Net Regularization Objective
To handle high-dimensional grouping and sparsity, we optimize the smooth Elastic-Net penalized negative log-partial likelihood using L-BFGS-B:

$$\mathcal{L}(\beta) = -\ell(\beta) + \alpha \left[ \rho \sum_{k=1}^{59} \sqrt{\beta_k^2 + \epsilon} + \frac{1-\rho}{2} \sum_{k=1}^{59} \beta_k^2 \right]$$

Where $\alpha$ controls penalty strength, $\rho \in [0, 1]$ is the $L_1 / L_2$ ratio, and $\epsilon = 10^{-6}$ provides smooth gradient differentiability at zero.
* **Figure 4 File:** `results/plots/fig4_smooth_l1_approximation.png`
* **Figure 5 File:** `results/plots/fig5_inference_xai_architecture.png`

#### 2.6 Isotonic Survival Probability Calibration (PAVA)
Relative risk scores $\eta_i$ are mapped into monotonic, un-biased survival probabilities $P(S > t \mid \eta_i)$ at horizons $t \in \{12, 24, 36, 60\}$ months using the Pool Adjacent Violators Algorithm (PAVA):

$$P(S > t \mid \eta_i) = 1 - f_{\text{iso}, t}(\eta_i), \quad \text{where } f_{\text{iso}, t} = \arg\min_{g \in \mathcal{M}} \sum_{i=1}^{N_{\text{cal}}} \left( y_{i,t} - g(\eta_i) \right)^2$$

Where $\mathcal{M}$ is the space of non-decreasing step functions fitted on an isolated calibration split ($20\%$).
* **Figure 6 File:** `results/plots/fig6_isotonic_calibration_curves.png`

#### 2.7 Inverse Transform Stochastic Monte Carlo Survival Simulation
We compute the cumulative baseline hazard $\Lambda_0(t)$ using the fitted model:

$$\Lambda_0(t) = \sum_{t_i \le t} \frac{d_i}{\sum_{j \in R(t_i)} \exp(\eta_j)}$$

For patient $i$, we sample 5,000 independent uniform random draws $U^{(k)} \sim \text{Uniform}(0, 1)$:

$$\Lambda_{\text{target}}^{(k)} = \frac{-\ln(U^{(k)})}{\exp(\eta_i)} \implies t^{(k)} = \Lambda_0^{-1}\left( \Lambda_{\text{target}}^{(k)} \right)$$

From the distribution $t^{(1 \dots 5000)}$, we extract non-parametric survival bounds:
* **$P10$ (Pessimistic):** 10th percentile survival time in months.
* **$P50$ (Median):** 50th percentile survival time in months.
* **$P90$ (Optimistic):** 90th percentile survival time in months.
* **Restricted Mean Survival Time (RMST):** $\text{RMST} = \frac{1}{5000} \sum_{k=1}^{5000} \min(t^{(k)}, 60.0)$.
* **Figure 7 File:** `results/plots/fig7_monte_carlo_trajectories.png`

#### 2.8 Closed-Form PCA Back-Projection & Local Patient Risk Waterfall
To break the PCA "black box", we substitute $X_{\text{pca}} = Z_{\text{gene}} V$ directly into the linear predictor formula:

$$\eta_{\text{pca}} = X_{\text{pca}} \beta_{\text{pca}} = (Z_{\text{gene}} V) \beta_{\text{pca}} = Z_{\text{gene}} (V \beta_{\text{pca}})$$

We define the **Global Gene Risk Weight Vector** $W_{\text{gene}} \in \mathbb{R}^{300}$:

$$\mathbf{W_{\text{gene}} = V_{300 \times 50} \cdot \beta_{\text{pca}} \in \mathbb{R}^{300}}$$

For individual patient $i$, the local risk contribution of raw gene $g$ is:

$$\Delta \eta_{g, i} = Z_{g, i} \cdot W_{\text{gene}, g}$$

Yielding an exact, closed-form additive risk waterfall:

$$\eta_i = \sum_{j=1}^{9} X_{i, j}^{\text{clin}} \beta_j^{\text{clin}} + \sum_{g=1}^{300} \Delta \eta_{g, i}$$

* **Figure 8 File:** `results/plots/fig8_global_gene_importance.png`
* **Figure 9 File:** `results/plots/fig9_patient_risk_waterfall.png`

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

### 📋 Master List of Standardized Image File Paths (`results/plots/`)

| Figure # | Exact PNG File Path on Disk | Description |
| :--- | :--- | :--- |
| **Figure 1** | `results/plots/fig1_pipeline_block_diagram.png` | End-to-end multi-omic survival analysis pipeline block diagram |
| **Figure 2** | `results/plots/fig2_component_io_block_diagram.png` | Component-level input/output matrix architecture diagram |
| **Figure 3** | `results/plots/fig3_gram_schmidt_collinearity.png` | Gram-Schmidt cosine collinearity pruning heatmap |
| **Figure 4** | `results/plots/fig4_smooth_l1_approximation.png` | Smooth $L_1$ penalty ($\sqrt{\beta^2 + 10^{-6}}$) differentiability curve |
| **Figure 5** | `results/plots/fig5_inference_xai_architecture.png` | Optimization & inference architectural data flow diagram |
| **Figure 6** | `results/plots/fig6_isotonic_calibration_curves.png` | Isotonic survival calibration reliability diagrams (PAVA) |
| **Figure 7** | `results/plots/fig7_monte_carlo_trajectories.png` | Monte Carlo 5,000-draw survival trajectories ($P10, P50, P90$) |
| **Figure 8** | `results/plots/fig8_global_gene_importance.png` | Global unrolled gene risk weights ($W_{\text{gene}}$) |
| **Figure 9** | `results/plots/fig9_patient_risk_waterfall.png` | Local patient additive risk waterfall decomposition |
