#!/usr/bin/env python3
"""
===============================================================================
FIGURE M1 / FIG 3: GRAM-SCHMIDT COSINE COLLINEARITY HEATMAP (NAN-FIXED)
===============================================================================
Generates side-by-side heatmaps illustrating pairwise gene-gene cosine similarity
before vs after Gram-Schmidt collinearity filtering (|CosSim| > 0.75 threshold).
Imputes NaNs before computing correlation to guarantee crisp, visible cells.
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
    print("🎨 Regenerating Fig 3: Gram-Schmidt Cosine Collinearity Heatmap (NaN Fixed)...")
    OUTPUT_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {CLEANED_DATA_PATH}")

    df = pd.read_parquet(CLEANED_DATA_PATH)

    # 12 representative genes containing tight collinear clusters
    collinear_genes = [
        "EXPR_FGA", "EXPR_FGB", "EXPR_FGG", 
        "EXPR_ORM1", "EXPR_ORM2", 
        "EXPR_ACSM2A", "EXPR_ACSM2B",
        "EXPR_CTRB1", "EXPR_CTRB2",
        "EXPR_APOA1", "EXPR_APOA2", "EXPR_BDK"
    ]
    selected_genes = [g for g in collinear_genes if g in df.columns]

    # Fill NaNs with 0.0 before log-transform and standardization
    X_raw = df[selected_genes].fillna(0.0).to_numpy(dtype=float)
    X_log = np.log2(X_raw + 1.0)
    X_std = (X_log - np.mean(X_log, axis=0)) / (np.std(X_log, axis=0) + 1e-8)

    # Compute correlation / cosine similarity matrix
    corr_matrix = np.abs(np.corrcoef(X_std, rowvar=False))
    
    all_gene_names = [g.replace("EXPR_", "") for g in selected_genes]
    cos_sim_before = corr_matrix

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

    # Plot Settings
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), dpi=300)

    # Subplot A: Before Filtering
    sns.heatmap(
        cos_sim_before,
        ax=axes[0],
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        cbar=True,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8.5, "weight": "bold"},
        linewidths=0.8,
        linecolor="white",
        xticklabels=all_gene_names,
        yticklabels=all_gene_names,
        square=True,
        cbar_kws={"shrink": 0.85}
    )
    axes[0].set_title(f"A) Pairwise Cosine Similarity (Before Filtering)\n{len(selected_genes)} High-Variance Genes Matrix", fontsize=13, fontweight="bold", pad=12)
    axes[0].tick_params(axis="x", rotation=45, labelsize=10.5)
    axes[0].tick_params(axis="y", rotation=0, labelsize=10.5)
    cbar0 = axes[0].collections[0].colorbar
    cbar0.ax.tick_params(labelsize=10.5)

    # Subplot B: After Gram-Schmidt Filtering
    sns.heatmap(
        cos_sim_after,
        ax=axes[1],
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        cbar=True,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9.5, "weight": "bold"},
        linewidths=0.8,
        linecolor="white",
        xticklabels=kept_gene_names,
        yticklabels=kept_gene_names,
        square=True,
        cbar_kws={"shrink": 0.85}
    )
    axes[1].set_title(f"B) Post Gram-Schmidt Subspace (|CosSim| ≤ 0.75)\n{len(kept_indices)} Retained Transcripts ({len(dropped_indices)} Pruned)", fontsize=13, fontweight="bold", pad=12)
    axes[1].tick_params(axis="x", rotation=45, labelsize=10.5)
    axes[1].tick_params(axis="y", rotation=0, labelsize=10.5)
    cbar1 = axes[1].collections[0].colorbar
    cbar1.ax.tick_params(labelsize=10.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, bbox_inches="tight")
    plt.savefig(ALT_OUTPUT_PATH, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved Fixed Fig 3 to: {OUTPUT_PLOT_PATH}")


if __name__ == "__main__":
    main()
