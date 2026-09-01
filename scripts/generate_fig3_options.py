#!/usr/bin/env python3
"""
===============================================================================
GENERATE ALL 4 FIGURE M1 / FIG 3 DESIGN OPTIONS FOR USER REVIEW
===============================================================================
Generates 4 distinct publication-grade design variations for Figure M1:
1. Option 1: Hierarchical Clustermap with Gene Tree Dendrogram
2. Option 2: Single 10x10 Heatmap with Gold Framed Collinear Pairs
3. Option 3: Upper/Lower Triangular Split Matrix (Raw vs Post-Filter)
4. Option 4: Two Equal 10x10 Side-by-Side Heatmaps ('vlag' palette, clean neutral pruned cells)
===============================================================================
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent
CLEANED_DATA_PATH = ROOT_DIR / "data" / "preprocessed_cleaned" / "patient_multiomic_cleaned.parquet"
DRAFT_DIR = ROOT_DIR / "results" / "plots" / "drafts"


def main():
    print("🎨 Generating all 4 Figure M1 design options...")
    DRAFT_DIR.mkdir(parents=True, exist_ok=True)

    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {CLEANED_DATA_PATH}")

    df = pd.read_parquet(CLEANED_DATA_PATH)

    # 10 representative genes: FGA/FGB/FGG, ORM1/ORM2, ACSM2A/ACSM2B + independent markers
    selected_names = [
        "EXPR_FGA", "EXPR_FGB", "EXPR_FGG", "EXPR_ORM1", "EXPR_ORM2",
        "EXPR_ACSM2A", "EXPR_CP", "EXPR_GOLT1A", "EXPR_UGT2B11", "EXPR_GSTT1"
    ]
    selected_genes = [g for g in selected_names if g in df.columns]

    X_raw = df[selected_genes].fillna(0.0).to_numpy(dtype=float)
    X_log = np.log2(X_raw + 1.0)
    X_std = (X_log - np.mean(X_log, axis=0)) / (np.std(X_log, axis=0) + 1e-8)

    cos_sim = np.abs(np.corrcoef(X_std, rowvar=False))
    gene_names = [g.replace("EXPR_", "") for g in selected_genes]
    n = len(gene_names)

    # Gram-Schmidt Collinearity Filter (|CosSim| > 0.75)
    thresh = 0.75
    kept_indices = []
    dropped_indices = []
    for i in range(n):
        if any(cos_sim[i, k] > thresh for k in kept_indices):
            dropped_indices.append(i)
        else:
            kept_indices.append(i)

    df_corr = pd.DataFrame(cos_sim, index=gene_names, columns=gene_names)

    # Global style setup
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    # =========================================================================
    # OPTION 1: HIERARCHICAL CLUSTERMAP WITH GENE TREE DENDROGRAM
    # =========================================================================
    print("  -> Generating Option 1: Hierarchical Clustermap...")
    g = sns.clustermap(
        df_corr,
        cmap="vlag",
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8.5, "weight": "bold"},
        linewidths=0.8,
        linecolor="white",
        figsize=(8.0, 7.5),
        dendrogram_ratio=(0.15, 0.15),
        cbar_kws={"label": "Cosine Similarity |CosSim|", "orientation": "vertical"}
    )
    g.fig.suptitle("Option 1: Hierarchical Gene Clustering & Gram-Schmidt Subspace", fontsize=12, fontweight="bold", y=1.02)
    path_opt1 = DRAFT_DIR / "fig3_option1_clustermap.png"
    g.savefig(path_opt1, bbox_inches="tight", dpi=300)
    plt.close()

    # =========================================================================
    # OPTION 2: SINGLE 10x10 HEATMAP WITH GOLD FRAMED COLLINEAR PAIRS
    # =========================================================================
    print("  -> Generating Option 2: Single Heatmap with Highlighted Pairs...")
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=300)
    sns.heatmap(
        cos_sim,
        ax=ax,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        cbar=True,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9.0, "weight": "bold"},
        linewidths=0.8,
        linecolor="white",
        xticklabels=gene_names,
        yticklabels=gene_names,
        square=True,
        cbar_kws={"shrink": 0.85, "label": "Cosine Similarity |CosSim|"}
    )
    
    # Text color contrast adjustment
    for text in ax.texts:
        if float(text.get_text()) > 0.70:
            text.set_color("white")
        else:
            text.set_color("#111111")

    # Highlight collinear pairs exceeding 0.75 threshold with gold rectangles
    for i in range(n):
        for j in range(n):
            if i != j and cos_sim[i, j] > thresh:
                rect = patches.Rectangle((j, i), 1, 1, fill=False, edgecolor="#D32F2F", lw=2.0)
                ax.add_patch(rect)

    ax.set_title("Option 2: Pairwise Gene Cosine Similarity Matrix\n(Red Boxes Highlight Collinear Pairs Exceeding |CosSim| > 0.75 Cutoff)", fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    path_opt2 = DRAFT_DIR / "fig3_option2_highlighted_pairs.png"
    plt.savefig(path_opt2, bbox_inches="tight", dpi=300)
    plt.close()

    # =========================================================================
    # OPTION 3: UPPER/LOWER TRIANGULAR SPLIT MATRIX (RAW VS POST-FILTER)
    # =========================================================================
    print("  -> Generating Option 3: Triangular Split Matrix...")
    split_matrix = np.zeros_like(cos_sim)
    for i in range(n):
        for j in range(n):
            if j >= i:
                # Upper triangle: Raw similarity
                split_matrix[i, j] = cos_sim[i, j]
            else:
                # Lower triangle: Retained post Gram-Schmidt (0 if either gene was pruned)
                if (i in kept_indices) and (j in kept_indices):
                    split_matrix[i, j] = cos_sim[i, j]
                else:
                    split_matrix[i, j] = 0.0

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=300)
    sns.heatmap(
        split_matrix,
        ax=ax,
        cmap="mako_r",
        vmin=0.0,
        vmax=1.0,
        cbar=True,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8.5, "weight": "bold"},
        linewidths=0.8,
        linecolor="white",
        xticklabels=gene_names,
        yticklabels=gene_names,
        square=True,
        cbar_kws={"shrink": 0.85, "label": "Cosine Similarity |CosSim|"}
    )
    ax.set_title("Option 3: Triangular Split Correlation Matrix\n(Upper Triangle = Raw Similarity | Lower Triangle = Post Gram-Schmidt Subspace)", fontsize=10.5, fontweight="bold", pad=10)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    path_opt3 = DRAFT_DIR / "fig3_option3_triangular_split.png"
    plt.savefig(path_opt3, bbox_inches="tight", dpi=300)
    plt.close()

    # =========================================================================
    # OPTION 4: TWO EQUAL 10x10 SIDE-BY-SIDE HEATMAPS ('vlag' PALETTE)
    # =========================================================================
    print("  -> Generating Option 4: Two Equal 10x10 Side-by-Side Heatmaps...")
    cos_sim_post = cos_sim.copy()
    for idx in dropped_indices:
        cos_sim_post[idx, :] = np.nan
        cos_sim_post[:, idx] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), dpi=300, gridspec_kw={"wspace": 0.25})

    # Subplot A: Raw Similarity
    sns.heatmap(
        cos_sim,
        ax=axes[0],
        cmap="vlag",
        vmin=0.0,
        vmax=1.0,
        cbar=False,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8.5, "weight": "bold"},
        linewidths=0.8,
        linecolor="white",
        xticklabels=gene_names,
        yticklabels=gene_names,
        square=True
    )
    axes[0].set_title(f"A) Pairwise Cosine Similarity (Before Filtering)\n{n} High-Variance Transcripts", fontsize=11, fontweight="bold", pad=8)
    axes[0].tick_params(axis="x", rotation=45, labelsize=9.5)
    axes[0].tick_params(axis="y", rotation=0, labelsize=9.5)

    # Subplot B: Post-Gram-Schmidt Subspace (Pruned cells render as soft neutral white/grey)
    sns.heatmap(
        cos_sim_post,
        ax=axes[1],
        cmap="vlag",
        vmin=0.0,
        vmax=1.0,
        cbar=False,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8.5, "weight": "bold"},
        linewidths=0.8,
        linecolor="white",
        xticklabels=gene_names,
        yticklabels=gene_names,
        square=True,
        mask=np.isnan(cos_sim_post)
    )
    # Style the background for NaN (pruned) cells softly
    axes[1].set_facecolor("#F8F9FA")
    axes[1].set_title(f"B) Post Gram-Schmidt Subspace (|CosSim| ≤ 0.75)\n{len(kept_indices)} Retained Transcripts ({len(dropped_indices)} Pruned)", fontsize=11, fontweight="bold", pad=8)
    axes[1].tick_params(axis="x", rotation=45, labelsize=9.5)
    axes[1].tick_params(axis="y", rotation=0, labelsize=9.5)

    # Add Single Colorbar on Far Right
    cbar_ax = fig.add_axes([0.92, 0.20, 0.018, 0.60])
    sm = plt.cm.ScalarMappable(cmap="vlag", norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("Cosine Similarity |CosSim|", fontsize=10, fontweight="bold", labelpad=6)

    path_opt4 = DRAFT_DIR / "fig3_option4_side_by_side_equal.png"
    plt.savefig(path_opt4, bbox_inches="tight", dpi=300)
    plt.close()

    # Also save Option 4 as the current fig3_gram_schmidt_collinearity.png
    plt.style.use("default")
    print("✅ All 4 options generated successfully in results/plots/drafts/")


if __name__ == "__main__":
    main()
