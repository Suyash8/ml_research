#!/usr/bin/env python3
"""
===============================================================================
FIGURE 3: OFFICIAL ORANGE COPPER GRADIENT HEATMAPS (BEFORE & AFTER)
===============================================================================
Generates side-by-side and individual heatmaps illustrating pairwise gene cosine similarity
before vs after Gram-Schmidt collinearity filtering (|CosSim| > 0.75 threshold).

Features:
  - Identical 8x8 matrix scale for 3A and 3B so cell sizes and aspect ratios match 100%.
  - 3B greys out dropped collinear features with clean 'DROP' markers.
  - Horizontal Cosine Similarity colorbars.
Outputs:
  - fig3_gram_schmidt_collinearity.png (combined)
  - fig3a_collinearity_before.png (standalone Subplot A)
  - fig3b_collinearity_after.png (standalone Subplot B)
===============================================================================
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent
CLEANED_DATA_PATH = ROOT_DIR / "data" / "preprocessed_cleaned" / "patient_multiomic_cleaned.parquet"
OUTPUT_PLOT_PATH = ROOT_DIR / "results" / "plots" / "fig3_gram_schmidt_collinearity.png"
ALT_OUTPUT_PATH = ROOT_DIR / "results" / "plots" / "figure_m1_gram_schmidt_collinearity.png"

OUTPUT_FIG3A = ROOT_DIR / "results" / "plots" / "fig3a_collinearity_before.png"
OUTPUT_FIG3B = ROOT_DIR / "results" / "plots" / "fig3b_collinearity_after.png"


def main():
    print("🎨 Generating Official Orange Copper Fig 3 Heatmaps (Perfect 8x8 Proportional Masking)...")
    OUTPUT_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {CLEANED_DATA_PATH}")

    df = pd.read_parquet(CLEANED_DATA_PATH)

    selected_names = [
        "EXPR_FGA", "EXPR_FGB", "EXPR_ORM1", "EXPR_ORM2", 
        "EXPR_CP", "EXPR_GOLT1A", "EXPR_UGT2B11", "EXPR_GSTT1"
    ]
    selected_genes = [g for g in selected_names if g in df.columns]

    X_raw = df[selected_genes].fillna(0.0).to_numpy(dtype=float)
    X_log = np.log2(X_raw + 1.0)
    X_std = (X_log - np.mean(X_log, axis=0)) / (np.std(X_log, axis=0) + 1e-8)

    cos_sim_before = np.abs(np.corrcoef(X_std, rowvar=False))
    all_gene_names = [g.replace("EXPR_", "") for g in selected_genes]

    thresh = 0.75
    kept_indices = []
    dropped_indices = []
    for i in range(len(selected_genes)):
        if any(cos_sim_before[i, k] > thresh for k in kept_indices):
            dropped_indices.append(i)
        else:
            kept_indices.append(i)

    white_to_copper = LinearSegmentedColormap.from_list(
        "WhiteToCopper", ["#FFFFFF", "#FFEDD5", "#FB923C", "#C2410C", "#7C2D12"], N=256
    )

    plt.style.use("seaborn-v0_8-white" if "seaborn-v0_8-white" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    # 1. Standalone Subplot A (fig3a_collinearity_before.png) - 8x8 Heatmap
    fig_a, ax_a = plt.subplots(figsize=(5.2, 5.2), dpi=300, facecolor="#FFFFFF")
    sns.heatmap(
        cos_sim_before, ax=ax_a, cmap=white_to_copper, vmin=0.0, vmax=1.0,
        cbar_kws={'orientation': 'horizontal', 'pad': 0.12, 'label': 'Cosine Similarity |CosSim|', 'shrink': 0.82},
        annot=True, fmt=".2f", annot_kws={"size": 8.0, "weight": "bold"},
        linewidths=1.0, linecolor="#CBD5E1", xticklabels=all_gene_names, yticklabels=all_gene_names, square=True
    )
    for text in ax_a.texts:
        val = float(text.get_text())
        text.set_color("#FFFFFF" if val > 0.65 else "#1E293B")
    ax_a.tick_params(axis="x", rotation=45, labelsize=9.5, colors="#334155")
    ax_a.tick_params(axis="y", rotation=0, labelsize=9.5, colors="#334155")
    plt.savefig(OUTPUT_FIG3A, bbox_inches="tight", facecolor="#FFFFFF", edgecolor="none")
    plt.close()

    # 2. Standalone Subplot B (fig3b_collinearity_after.png) - 8x8 Masked Heatmap (Matching Scale & Size!)
    cos_sim_masked = cos_sim_before.copy()
    mask_matrix = np.zeros_like(cos_sim_masked, dtype=bool)
    for idx in dropped_indices:
        mask_matrix[idx, :] = True
        mask_matrix[:, idx] = True

    fig_b, ax_b = plt.subplots(figsize=(5.2, 5.2), dpi=300, facecolor="#FFFFFF")
    sns.heatmap(
        np.ones_like(cos_sim_masked), ax=ax_b, cmap=LinearSegmentedColormap.from_list("Grey", ["#F1F5F9", "#F1F5F9"]),
        cbar=False, annot=False, linewidths=1.0, linecolor="#CBD5E1", xticklabels=all_gene_names, yticklabels=all_gene_names, square=True
    )
    sns.heatmap(
        cos_sim_masked, ax=ax_b, mask=mask_matrix, cmap=white_to_copper, vmin=0.0, vmax=1.0,
        cbar_kws={'orientation': 'horizontal', 'pad': 0.12, 'label': 'Cosine Similarity |CosSim| (|r| ≤ 0.75)', 'shrink': 0.82},
        annot=True, fmt=".2f", annot_kws={"size": 8.0, "weight": "bold"},
        linewidths=1.0, linecolor="#CBD5E1", xticklabels=all_gene_names, yticklabels=all_gene_names, square=True
    )
    for text in ax_b.texts:
        if text.get_text():
            val = float(text.get_text())
            text.set_color("#FFFFFF" if val > 0.65 else "#1E293B")

    for idx in dropped_indices:
        ax_b.text(idx + 0.5, idx + 0.5, "DROP", ha="center", va="center", color="#94A3B8", fontsize=8.0, fontweight="bold")

    ax_b.tick_params(axis="x", rotation=45, labelsize=9.5, colors="#334155")
    ax_b.tick_params(axis="y", rotation=0, labelsize=9.5, colors="#334155")
    plt.savefig(OUTPUT_FIG3B, bbox_inches="tight", facecolor="#FFFFFF", edgecolor="none")
    plt.close()

    # 3. Combined Figure 3
    fig = plt.figure(figsize=(11.0, 5.2), dpi=300, facecolor="#FFFFFF")
    fig.subplots_adjust(left=0.06, right=0.90, top=0.88, bottom=0.15, wspace=0.28)
    gs = fig.add_gridspec(1, 2)
    ax0 = fig.add_subplot(gs[0], facecolor="#FFFFFF")
    ax1 = fig.add_subplot(gs[1], facecolor="#FFFFFF")

    sns.heatmap(
        cos_sim_before, ax=ax0, cmap=white_to_copper, vmin=0.0, vmax=1.0, cbar=False,
        annot=True, fmt=".2f", annot_kws={"size": 8.0, "weight": "bold"},
        linewidths=1.0, linecolor="#CBD5E1", xticklabels=all_gene_names, yticklabels=all_gene_names, square=True
    )
    for text in ax0.texts:
        val = float(text.get_text())
        text.set_color("#FFFFFF" if val > 0.65 else "#1E293B")
    ax0.set_title("A) Pairwise Cosine Similarity (Before Filtering)", fontsize=11.0, fontweight="bold", color="#0F172A", pad=10)

    sns.heatmap(
        np.ones_like(cos_sim_masked), ax=ax1, cmap=LinearSegmentedColormap.from_list("Grey", ["#F1F5F9", "#F1F5F9"]),
        cbar=False, annot=False, linewidths=1.0, linecolor="#CBD5E1", xticklabels=all_gene_names, yticklabels=all_gene_names, square=True
    )
    sns.heatmap(
        cos_sim_masked, ax=ax1, mask=mask_matrix, cmap=white_to_copper, vmin=0.0, vmax=1.0, cbar=False,
        annot=True, fmt=".2f", annot_kws={"size": 8.0, "weight": "bold"},
        linewidths=1.0, linecolor="#CBD5E1", xticklabels=all_gene_names, yticklabels=all_gene_names, square=True
    )
    for text in ax1.texts:
        if text.get_text():
            val = float(text.get_text())
            text.set_color("#FFFFFF" if val > 0.65 else "#1E293B")
    for idx in dropped_indices:
        ax1.text(idx + 0.5, idx + 0.5, "DROP", ha="center", va="center", color="#94A3B8", fontsize=8.0, fontweight="bold")

    ax1.set_title(f"B) Post Gram-Schmidt Subspace (|CosSim| ≤ {thresh})", fontsize=11.0, fontweight="bold", color="#0F172A", pad=10)

    plt.savefig(OUTPUT_PLOT_PATH, bbox_inches="tight", facecolor="#FFFFFF", edgecolor="none")
    plt.savefig(ALT_OUTPUT_PATH, bbox_inches="tight", facecolor="#FFFFFF", edgecolor="none")
    plt.close()

    print(f"✅ Saved perfectly matched 8x8 Fig 3, Fig 3a ({OUTPUT_FIG3A.name}), and Fig 3b ({OUTPUT_FIG3B.name})")


if __name__ == "__main__":
    main()
