#!/usr/bin/env python3
"""
===============================================================================
FIGURE 4: PRINCIPAL COMPONENT ANALYSIS LATENT GENE LOADING MAP
===============================================================================
Generates a publication-ready heatmap displaying the top gene loadings V across
the principal components (PC01-PC10), illustrating how raw transcripts map onto
latent PCA dimensions.
Outputs:
  - results/plots/fig4_pca_gene_loadings.png
  - images/fig4_pca_gene_loadings.png
  - docs/images/fig4_pca_gene_loadings.png
===============================================================================
"""

from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent
LOADINGS_CSV = ROOT_DIR / "results" / "explainability" / "xai_pca_component_loadings.csv"
OUTPUT_PLOT_PATH = ROOT_DIR / "results" / "plots" / "fig4_pca_gene_loadings.png"
IMAGES_PLOT_PATH = ROOT_DIR / "images" / "fig4_pca_gene_loadings.png"
DOCS_IMAGES_PATH = ROOT_DIR / "docs" / "images" / "fig4_pca_gene_loadings.png"

def main():
    print("🎨 Generating Figure 4: PCA Gene Loadings Map (Clean PC01-PC10 Labels, Zero Overlap)...")
    OUTPUT_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    IMAGES_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCS_IMAGES_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not LOADINGS_CSV.exists():
        raise FileNotFoundError(f"Loadings CSV not found at {LOADINGS_CSV}")

    df = pd.read_csv(LOADINGS_CSV)
    
    # Filter for top 10 PCs and top 20 genes with highest loading variance
    top_pcs = df[df["pc_index"] <= 10].copy()
    top_genes = top_pcs.groupby("gene_name")["gene_loading"].apply(lambda x: np.max(np.abs(x))).nlargest(20).index
    
    heatmap_df = top_pcs[top_pcs["gene_name"].isin(top_genes)].pivot(
        index="gene_name", columns="pc_name", values="gene_loading"
    ).fillna(0.0)

    # Clean row and column names (strip "EXPR_" prefix)
    heatmap_df.index = [g.replace("EXPR_", "") for g in heatmap_df.index]
    heatmap_df.columns = [c.replace("EXPR_", "") for c in heatmap_df.columns]

    plt.style.use("seaborn-v0_8-white" if "seaborn-v0_8-white" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=300, facecolor="#FFFFFF")

    # Blue-White-Orange Diverging Colormap
    cmap = LinearSegmentedColormap.from_list(
        "LoadingDiverging", ["#0284C7", "#F8FAFC", "#EA580C"], N=256
    )

    sns.heatmap(
        heatmap_df,
        ax=ax,
        cmap=cmap,
        center=0.0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8.5, "weight": "bold"},
        linewidths=1.2,
        linecolor="#E2E8F0",
        cbar_kws={"label": "Eigenvector Loading V [300 × 50]", "pad": 0.02},
    )

    ax.set_title("Principal Component Analysis Latent Gene Loading Map (Top Transcripts vs PCs)", fontsize=12, fontweight="bold", color="#0F172A", pad=12)
    ax.set_xlabel("Principal Component Subspace", fontsize=11, fontweight="bold", color="#334155", labelpad=8)
    ax.set_ylabel("High-Variance Gene Transcripts", fontsize=11, fontweight="bold", color="#334155", labelpad=8)
    
    # Clean x and y ticks with ample spacing and ZERO overlap
    ax.tick_params(axis="x", rotation=0, labelsize=10, colors="#334155", pad=4)
    ax.tick_params(axis="y", rotation=0, labelsize=9.5, colors="#334155", pad=4)

    plt.savefig(OUTPUT_PLOT_PATH, bbox_inches="tight", facecolor="#FFFFFF", edgecolor="none")
    plt.savefig(IMAGES_PLOT_PATH, bbox_inches="tight", facecolor="#FFFFFF", edgecolor="none")
    shutil.copy(OUTPUT_PLOT_PATH, DOCS_IMAGES_PATH)
    plt.close()

    print(f"✅ Saved Fig 4 PCA Loadings with zero overlap to: {OUTPUT_PLOT_PATH}, {IMAGES_PLOT_PATH}, and {DOCS_IMAGES_PATH}")

if __name__ == "__main__":
    main()
