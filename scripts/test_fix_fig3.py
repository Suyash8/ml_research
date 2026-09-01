#!/usr/bin/env python3
"""
Fix fig3a and fig3b heatmaps with clean text padding, proportional cell sizes,
and no text overlap.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent
CLEANED_DATA_PATH = ROOT_DIR / "data" / "preprocessed_cleaned" / "patient_multiomic_cleaned.parquet"
OUTPUT_FIG3A = ROOT_DIR / "results" / "plots" / "fig3a_collinearity_before.png"
OUTPUT_FIG3B = ROOT_DIR / "results" / "plots" / "fig3b_collinearity_after.png"

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
for i in range(len(selected_genes)):
    if not any(cos_sim_before[i, k] > thresh for k in kept_indices):
        kept_indices.append(i)

cos_sim_after = cos_sim_before[np.ix_(kept_indices, kept_indices)]
kept_gene_names = [selected_genes[i].replace("EXPR_", "") for i in kept_indices]

white_to_copper = LinearSegmentedColormap.from_list(
    "WhiteToCopper", ["#FFFFFF", "#FFEDD5", "#FB923C", "#C2410C", "#7C2D12"], N=256
)

plt.style.use("seaborn-v0_8-white" if "seaborn-v0_8-white" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

# 1. FIG 3A (Before) - Clean text font size 8.5pt bold, crisp margins
fig_a, ax_a = plt.subplots(figsize=(5.5, 5.8), dpi=300, facecolor="#FFFFFF")
sns.heatmap(
    cos_sim_before, ax=ax_a, cmap=white_to_copper, vmin=0.0, vmax=1.0,
    cbar_kws={'orientation': 'horizontal', 'pad': 0.12, 'label': 'Cosine Similarity |CosSim|', 'shrink': 0.85},
    annot=True, fmt=".2f", annot_kws={"size": 8.2, "weight": "bold"},
    linewidths=1.2, linecolor="#FFFFFF", xticklabels=all_gene_names, yticklabels=all_gene_names, square=True
)
for text in ax_a.texts:
    val = float(text.get_text())
    text.set_color("#FFFFFF" if val > 0.65 else "#1E293B")

ax_a.tick_params(axis="x", rotation=45, labelsize=9.5, colors="#334155")
ax_a.tick_params(axis="y", rotation=0, labelsize=9.5, colors="#334155")
plt.savefig(OUTPUT_FIG3A, bbox_inches="tight", facecolor="#FFFFFF", edgecolor="none")
plt.close()

# 2. FIG 3B (After) - Scaled figsize proportional to number of kept genes so cell size matches Fig 3A!
n_kept = len(kept_gene_names)
# 8 genes in 3A took ~4.2 inch width. So n_kept genes should take (4.2 * n_kept / 8) inch width!
width_b = 5.5 * (n_kept / 8.0) + 1.2
fig_b, ax_b = plt.subplots(figsize=(width_b, 5.8), dpi=300, facecolor="#FFFFFF")
sns.heatmap(
    cos_sim_after, ax=ax_b, cmap=white_to_copper, vmin=0.0, vmax=1.0,
    cbar_kws={'orientation': 'horizontal', 'pad': 0.12, 'label': 'Cosine Similarity |CosSim|', 'shrink': 0.85},
    annot=True, fmt=".2f", annot_kws={"size": 8.2, "weight": "bold"},
    linewidths=1.2, linecolor="#FFFFFF", xticklabels=kept_gene_names, yticklabels=kept_gene_names, square=True
)
for text in ax_b.texts:
    val = float(text.get_text())
    text.set_color("#FFFFFF" if val > 0.65 else "#1E293B")

ax_b.tick_params(axis="x", rotation=45, labelsize=9.5, colors="#334155")
ax_b.tick_params(axis="y", rotation=0, labelsize=9.5, colors="#334155")
plt.savefig(OUTPUT_FIG3B, bbox_inches="tight", facecolor="#FFFFFF", edgecolor="none")
plt.close()

print(f"Generated test fix for 3A and 3B: n_kept={n_kept}")
