#!/usr/bin/env python3
"""
===============================================================================
FIGURE M1 / FIG 3: GRAM-SCHMIDT COSINE COLLINEARITY HEATMAP (LARGE FONTS)
===============================================================================
Generates side-by-side heatmaps illustrating pairwise gene-gene cosine similarity
before vs after Gram-Schmidt collinearity filtering (|CosSim| > 0.75 threshold).
Enforces large, highly legible fonts matching LaTeX document publication standards.
===============================================================================
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent
CLEANED_DATA_PATH = ROOT_DIR / "data" / "preprocessed_cleaned" / "patient_multiomic_cleaned.parquet"
OUTPUT_PLOT_PATH = ROOT_DIR / "results" / "plots" / "fig3_gram_schmidt_collinearity.png"
ALT_OUTPUT_PATH = ROOT_DIR / "results" / "plots" / "figure_m1_gram_schmidt_collinearity.png"


def main():
    print("🎨 Regenerating Fig 3: Gram-Schmidt Cosine Collinearity Heatmap (Large Fonts)...")
    OUTPUT_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {CLEANED_DATA_PATH}")

    df = pd.read_parquet(CLEANED_DATA_PATH)
    gene_cols = [c for c in df.columns if c.startswith("EXPR_")]

    # Select high-variance genes for correlation matrix
    variances = df[gene_cols].var().sort_values(ascending=False)
    candidate_genes = variances.head(50).index.tolist()

    X_raw = df[candidate_genes].to_numpy(dtype=float)
    X_log = np.log2(X_raw + 1.0)
    X_std = (X_log - np.mean(X_log, axis=0)) / (np.std(X_log, axis=0) + 1e-8)

    corr_matrix = np.abs(np.corrcoef(X_std, rowvar=False))
    
    selected_indices = list(range(18))
    selected_genes = [candidate_genes[i] for i in selected_indices]
    cos_sim_before = corr_matrix[np.ix_(selected_indices, selected_indices)]

    # Apply Gram-Schmidt Filter (|CosSim| > 0.70 threshold)
    kept_indices_sub = []
    dropped_indices_sub = []
    for i in range(len(selected_indices)):
        is_collinear = False
        for k in kept_indices_sub:
            if cos_sim_before[i, k] > 0.70:
                is_collinear = True
                break
        if not is_collinear:
            kept_indices_sub.append(i)
        else:
            dropped_indices_sub.append(i)

    cos_sim_after = cos_sim_before[np.ix_(kept_indices_sub, kept_indices_sub)]
    kept_gene_names = [selected_genes[i].replace("EXPR_", "") for i in kept_indices_sub]
    all_gene_names = [g.replace("EXPR_", "") for g in selected_genes]

    # Large Font Settings for LaTeX Readability
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=300)

    # Subplot A: Before
    sns.heatmap(
        cos_sim_before,
        ax=axes[0],
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        cbar=True,
        xticklabels=all_gene_names,
        yticklabels=all_gene_names,
        square=True,
        cbar_kws={"shrink": 0.82}
    )
    axes[0].set_title(f"A) Before Filtering ({len(selected_genes)} High-Variance Genes)", fontsize=13, fontweight="bold", pad=10)
    axes[0].tick_params(axis="x", rotation=90, labelsize=9.5)
    axes[0].tick_params(axis="y", rotation=0, labelsize=9.5)
    cbar0 = axes[0].collections[0].colorbar
    cbar0.ax.tick_params(labelsize=10)

    # Subplot B: After
    sns.heatmap(
        cos_sim_after,
        ax=axes[1],
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        cbar=True,
        xticklabels=kept_gene_names,
        yticklabels=kept_gene_names,
        square=True,
        cbar_kws={"shrink": 0.82}
    )
    axes[1].set_title(f"B) Post Gram-Schmidt Subspace ({len(kept_indices_sub)} Kept, {len(dropped_indices_sub)} Pruned)", fontsize=13, fontweight="bold", pad=10)
    axes[1].tick_params(axis="x", rotation=90, labelsize=9.5)
    axes[1].tick_params(axis="y", rotation=0, labelsize=9.5)
    cbar1 = axes[1].collections[0].colorbar
    cbar1.ax.tick_params(labelsize=10)

    # NO suptitle to avoid double header in LaTeX
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, bbox_inches="tight")
    plt.savefig(ALT_OUTPUT_PATH, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved Fig 3 to: {OUTPUT_PLOT_PATH}")


if __name__ == "__main__":
    main()
