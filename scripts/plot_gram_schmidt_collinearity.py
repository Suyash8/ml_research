#!/usr/bin/env python3
"""
===============================================================================
FIGURE M1 / FIG 3: ELEGANT PUBLICATION-GRADE GRAM-SCHMIDT HEATMAP
===============================================================================
Generates side-by-side balanced heatmaps illustrating pairwise gene cosine similarity
before vs after Gram-Schmidt collinearity filtering (|CosSim| > 0.75 threshold).
Enforces proportional matrix sizes, crisp text contrast, and a single shared colorbar.
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
    print("🎨 Regenerating Fig 3: Elegant Gram-Schmidt Collinearity Heatmap...")
    OUTPUT_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {CLEANED_DATA_PATH}")

    df = pd.read_parquet(CLEANED_DATA_PATH)

    # Pick 16 genes with a mix of highly correlated pairs and independent genes
    candidate_genes = [
        "EXPR_FGA", "EXPR_FGB", "EXPR_FGG",
        "EXPR_ORM1", "EXPR_ORM2",
        "EXPR_ACSM2A", "EXPR_ACSM2B",
        "EXPR_CTRB1", "EXPR_CTRB2",
        "EXPR_APOA1", "EXPR_APOA2",
        "EXPR_BDK", "EXPR_AHSG", "EXPR_HRG", "EXPR_HP", "EXPR_PLG"
    ]
    selected_genes = [g for g in candidate_genes if g in df.columns]

    X_raw = df[selected_genes].fillna(0.0).to_numpy(dtype=float)
    X_log = np.log2(X_raw + 1.0)
    X_std = (X_log - np.mean(X_log, axis=0)) / (np.std(X_log, axis=0) + 1e-8)

    # Pairwise correlation / cosine similarity
    cos_sim_before = np.abs(np.corrcoef(X_std, rowvar=False))
    all_gene_names = [g.replace("EXPR_", "") for g in selected_genes]

    # Apply Gram-Schmidt Collinearity Pruning (|CosSim| > 0.75 threshold)
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

    # Style configuration matching Figures M2, M3, M4
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), dpi=300, gridspec_kw={"wspace": 0.28})

    # Subplot A: Before Filtering (Full 16x16 Matrix)
    im0 = axes[0].imshow(cos_sim_before, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
    axes[0].set_title(f"A) Pairwise Cosine Similarity (Before Filtering)\n{len(selected_genes)} High-Variance Genes Matrix", fontsize=12, fontweight="bold", pad=10)
    axes[0].set_xticks(range(len(all_gene_names)))
    axes[0].set_yticks(range(len(all_gene_names)))
    axes[0].set_xticklabels(all_gene_names, rotation=45, ha="right", fontsize=9.5, fontweight="bold")
    axes[0].set_yticklabels(all_gene_names, fontsize=9.5, fontweight="bold")
    axes[0].grid(False)

    # Subplot B: After Gram-Schmidt Filtering (11x11 Retained Matrix - Balanced Proportion)
    im1 = axes[1].imshow(cos_sim_after, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
    axes[1].set_title(f"B) Post Gram-Schmidt Subspace (|CosSim| ≤ 0.75)\n{len(kept_indices)} Retained Transcripts ({len(dropped_indices)} Pruned)", fontsize=12, fontweight="bold", pad=10)
    axes[1].set_xticks(range(len(kept_gene_names)))
    axes[1].set_yticks(range(len(kept_gene_names)))
    axes[1].set_xticklabels(kept_gene_names, rotation=45, ha="right", fontsize=9.5, fontweight="bold")
    axes[1].set_yticklabels(kept_gene_names, fontsize=9.5, fontweight="bold")
    axes[1].grid(False)

    # Add numeric text labels with contrast-aware text coloring
    for i in range(len(all_gene_names)):
        for j in range(len(all_gene_names)):
            val = cos_sim_before[i, j]
            text_color = "white" if val > 0.65 else "black"
            axes[0].text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=7.0, fontweight="bold")

    for i in range(len(kept_gene_names)):
        for j in range(len(kept_gene_names)):
            val = cos_sim_after[i, j]
            text_color = "white" if val > 0.65 else "black"
            axes[1].text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8.0, fontweight="bold")

    # Add single elegant colorbar on right
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.82, pad=0.02)
    cbar.ax.tick_params(labelsize=10.5)
    cbar.set_label("Cosine Similarity |CosSim|", fontsize=11, fontweight="bold", labelpad=8)

    plt.savefig(OUTPUT_PLOT_PATH, bbox_inches="tight")
    plt.savefig(ALT_OUTPUT_PATH, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved Publication-Grade Fig 3 to: {OUTPUT_PLOT_PATH}")


if __name__ == "__main__":
    main()
