#!/usr/bin/env python3
"""
===============================================================================
FIGURE M1: GRAM-SCHMIDT COSINE COLLINEARITY PRUNING HEATMAP
===============================================================================
Generates side-by-side heatmaps illustrating pairwise gene-gene cosine similarity
before vs after Gram-Schmidt collinearity filtering (|CosSim| > 0.75 threshold).
===============================================================================
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent
CLEANED_DATA_PATH = ROOT_DIR / "data" / "preprocessed_cleaned" / "patient_multiomic_cleaned.parquet"
OUTPUT_PLOT_PATH = ROOT_DIR / "results" / "plots" / "figure_m1_gram_schmidt_collinearity.png"


def main():
    print("🎨 Generating Figure M1: Gram-Schmidt Cosine Collinearity Heatmap...")
    OUTPUT_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {CLEANED_DATA_PATH}")

    df = pd.read_parquet(CLEANED_DATA_PATH)
    gene_cols = [c for c in df.columns if c.startswith("EXPR_")]

    # Pick top 25 high-variance genes for visual clarity in heatmap
    variances = df[gene_cols].var().sort_values(ascending=False)
    selected_genes = variances.head(25).index.tolist()

    X_genes = df[selected_genes].to_numpy(dtype=float)
    
    # Compute Cosine Similarity Matrix: CosSim(a, b) = <a,b> / (||a||*||b||)
    norms = np.linalg.norm(X_genes, axis=0, keepdims=True)
    norms[norms == 0] = 1e-12
    X_norm = X_genes / norms
    cos_sim_before = np.abs(np.dot(X_norm.T, X_norm))

    # Apply Gram-Schmidt Filter (|CosSim| > 0.75 threshold)
    kept_indices = []
    dropped_indices = []
    for i in range(len(selected_genes)):
        is_collinear = False
        for k in kept_indices:
            if cos_sim_before[i, k] > 0.75:
                is_collinear = True
                break
        if not is_collinear:
            kept_indices.append(i)
        else:
            dropped_indices.append(i)

    cos_sim_after = cos_sim_before[np.ix_(kept_indices, kept_indices)]
    kept_gene_names = [selected_genes[i].replace("EXPR_", "") for i in kept_indices]
    all_gene_names = [g.replace("EXPR_", "") for g in selected_genes]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Heatmap Before
    sns.heatmap(
        cos_sim_before,
        ax=axes[0],
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        cbar=True,
        xticklabels=all_gene_names,
        yticklabels=all_gene_names,
        square=True
    )
    axes[0].set_title(f"A) Pairwise Cosine Similarity (Before Filtering)\nTop {len(selected_genes)} High-Variance Genes", fontsize=11, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=90, labelsize=7)
    axes[0].tick_params(axis="y", rotation=0, labelsize=7)

    # Heatmap After
    sns.heatmap(
        cos_sim_after,
        ax=axes[1],
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        cbar=True,
        xticklabels=kept_gene_names,
        yticklabels=kept_gene_names,
        square=True
    )
    axes[1].set_title(f"B) Post Gram-Schmidt Subspace (|CosSim| ≤ 0.75)\n{len(kept_indices)} Non-Redundant Transcripts Retained ({len(dropped_indices)} Pruned)", fontsize=11, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=90, labelsize=7)
    axes[1].tick_params(axis="y", rotation=0, labelsize=7)

    plt.suptitle("Figure M1: Gram-Schmidt Cosine Collinearity Pruning Subspace", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved Figure M1 to: {OUTPUT_PLOT_PATH}")


if __name__ == "__main__":
    main()
