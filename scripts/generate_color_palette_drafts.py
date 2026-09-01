#!/usr/bin/env python3
"""
===============================================================================
GENERATE ALL SINGLE-HUE COLOR PALETTE DRAFTS FOR FIGURE M1
===============================================================================
Generates 7 single-hue white-background heatmaps:
1. Deep Navy Blue
2. Emerald Teal / Green
3. Royal Purple
4. Crimson / Burgundy Red
5. Slate Charcoal / Grey
6. Copper / Dark Amber
7. Seaborn Crest (Teal-Green)
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
DRAFT_DIR = ROOT_DIR / "results" / "plots" / "drafts"


def generate_single_hue_plot(df, selected_genes, cmap_obj, dark_text_color, filename, title_color_name):
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

    cos_sim_after = cos_sim_before[np.ix_(kept_indices, kept_indices)]
    kept_gene_names = [selected_genes[i].replace("EXPR_", "") for i in kept_indices]

    plt.style.use("seaborn-v0_8-white" if "seaborn-v0_8-white" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    fig = plt.figure(figsize=(11.5, 4.8), dpi=300, facecolor="#FFFFFF")
    fig.subplots_adjust(left=0.06, right=0.90, top=0.88, bottom=0.15, wspace=0.28)

    gs = fig.add_gridspec(1, 2, width_ratios=[len(all_gene_names), len(kept_gene_names)])
    ax0 = fig.add_subplot(gs[0], facecolor="#FFFFFF")
    ax1 = fig.add_subplot(gs[1], facecolor="#FFFFFF")

    # Subplot A
    sns.heatmap(
        cos_sim_before,
        ax=ax0,
        cmap=cmap_obj,
        vmin=0.0,
        vmax=1.0,
        cbar=False,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9.0, "weight": "bold"},
        linewidths=1.0,
        linecolor="#E5E7EB",
        xticklabels=all_gene_names,
        yticklabels=all_gene_names,
        square=True
    )
    for text in ax0.texts:
        val = float(text.get_text())
        if val > 0.65:
            text.set_color("#FFFFFF")
        else:
            text.set_color("#1E293B")

    ax0.set_title(f"A) Raw Similarity Matrix ({title_color_name})", fontsize=11.5, fontweight="bold", color="#0F172A", pad=10)
    ax0.tick_params(axis="x", rotation=45, labelsize=10, colors="#334155")
    ax0.tick_params(axis="y", rotation=0, labelsize=10, colors="#334155")

    # Subplot B
    sns.heatmap(
        cos_sim_after,
        ax=ax1,
        cmap=cmap_obj,
        vmin=0.0,
        vmax=1.0,
        cbar=False,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9.5, "weight": "bold"},
        linewidths=1.0,
        linecolor="#E5E7EB",
        xticklabels=kept_gene_names,
        yticklabels=kept_gene_names,
        square=True
    )
    for text in ax1.texts:
        val = float(text.get_text())
        if val > 0.65:
            text.set_color("#FFFFFF")
        else:
            text.set_color("#1E293B")

    ax1.set_title(f"B) Post Gram-Schmidt Subspace (|CosSim| ≤ {thresh})", fontsize=11.5, fontweight="bold", color="#0F172A", pad=10)
    ax1.tick_params(axis="x", rotation=45, labelsize=10, colors="#334155")
    ax1.tick_params(axis="y", rotation=0, labelsize=10, colors="#334155")

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.18, 0.016, 0.65], facecolor="#FFFFFF")
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=9.5, colors="#334155")
    cbar.set_label("Cosine Similarity |CosSim|", fontsize=10.5, fontweight="bold", color="#0F172A", labelpad=8)
    cbar.outline.set_edgecolor("#CBD5E1")
    cbar.outline.set_linewidth(1.0)

    out_path = DRAFT_DIR / filename
    plt.savefig(out_path, bbox_inches="tight", facecolor="#FFFFFF", edgecolor="none")
    plt.close()
    print(f"  -> Generated {filename}")


def main():
    print("🎨 Generating all single-hue color palette drafts...")
    DRAFT_DIR.mkdir(parents=True, exist_ok=True)

    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {CLEANED_DATA_PATH}")

    df = pd.read_parquet(CLEANED_DATA_PATH)
    selected_names = [
        "EXPR_FGA", "EXPR_FGB", "EXPR_ORM1", "EXPR_ORM2", 
        "EXPR_CP", "EXPR_GOLT1A", "EXPR_UGT2B11", "EXPR_GSTT1"
    ]
    selected_genes = [g for g in selected_names if g in df.columns]

    # Palettes dictionary
    palettes = {
        "fig3_color1_navy.png": (
            LinearSegmentedColormap.from_list("Navy", ["#FFFFFF", "#E8EEF5", "#8FAADC", "#2F5597", "#0F2C59"], N=256),
            "#0F2C59",
            "Deep Navy Blue"
        ),
        "fig3_color2_teal.png": (
            LinearSegmentedColormap.from_list("Teal", ["#FFFFFF", "#E6F4F1", "#80C7B9", "#148F77", "#064E3B"], N=256),
            "#064E3B",
            "Emerald Teal"
        ),
        "fig3_color3_purple.png": (
            LinearSegmentedColormap.from_list("Purple", ["#FFFFFF", "#F3E8FF", "#C084FC", "#7E22CE", "#4C1D95"], N=256),
            "#4C1D95",
            "Royal Purple"
        ),
        "fig3_color4_burgundy.png": (
            LinearSegmentedColormap.from_list("Burgundy", ["#FFFFFF", "#FFE4E6", "#FB7185", "#BE123C", "#700B2B"], N=256),
            "#700B2B",
            "Burgundy Red"
        ),
        "fig3_color5_charcoal.png": (
            LinearSegmentedColormap.from_list("Charcoal", ["#FFFFFF", "#F1F5F9", "#94A3B8", "#475569", "#0F172A"], N=256),
            "#0F172A",
            "Slate Charcoal"
        ),
        "fig3_color6_copper.png": (
            LinearSegmentedColormap.from_list("Copper", ["#FFFFFF", "#FFEDD5", "#FB923C", "#C2410C", "#7C2D12"], N=256),
            "#7C2D12",
            "Copper / Amber"
        ),
        "fig3_color7_crest.png": (
            sns.color_palette("crest", as_cmap=True),
            "#134E4A",
            "Seaborn Crest"
        )
    }

    for filename, (cmap_obj, dark_text, title_name) in palettes.items():
        generate_single_hue_plot(df, selected_genes, cmap_obj, dark_text, filename, title_name)

    print("✅ All 7 single-hue color palette drafts generated successfully!")


if __name__ == "__main__":
    main()
