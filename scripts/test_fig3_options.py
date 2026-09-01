#!/usr/bin/env python3
"""
Test Option 1 (8x8 with Pruned/Greyed cells) vs Option 2 (6x6 matrix with proportional scaling).
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
OUTPUT_FIG3B_OPT1 = ROOT_DIR / "results" / "plots" / "fig3b_opt1_masked.png"
OUTPUT_FIG3B_OPT2 = ROOT_DIR / "results" / "plots" / "fig3b_opt2_6x6.png"

df = pd.read_parquet(CLEANED_DATA_PATH)

# Option 1: 8 genes, prune high correlations
selected_names_8 = [
    "EXPR_FGA", "EXPR_FGB", "EXPR_ORM1", "EXPR_ORM2", 
    "EXPR_CP", "EXPR_GOLT1A", "EXPR_UGT2B11", "EXPR_GSTT1"
]
selected_genes = [g for g in selected_names_8 if g in df.columns]

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

# Clean 3A (Before)
fig_a, ax_a = plt.subplots(figsize=(5.2, 5.2), dpi=300, facecolor="#FFFFFF")
sns.heatmap(
    cos_sim_before, ax=ax_a, cmap=white_to_copper, vmin=0.0, vmax=1.0,
    cbar_kws={'orientation': 'horizontal', 'pad': 0.12, 'label': 'Cosine Similarity |CosSim|', 'shrink': 0.8},
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

# Option 1: 8x8 Masked (Dropped features greyed out)
cos_sim_masked = cos_sim_before.copy()
mask_matrix = np.zeros_like(cos_sim_masked, dtype=bool)
for idx in dropped_indices:
    mask_matrix[idx, :] = True
    mask_matrix[:, idx] = True

fig_b1, ax_b1 = plt.subplots(figsize=(5.2, 5.2), dpi=300, facecolor="#FFFFFF")
# Draw base grey for dropped features
sns.heatmap(
    np.ones_like(cos_sim_masked), ax=ax_b1, cmap=LinearSegmentedColormap.from_list("Grey", ["#F1F5F9", "#F1F5F9"]),
    cbar=False, annot=False, linewidths=1.0, linecolor="#CBD5E1", xticklabels=all_gene_names, yticklabels=all_gene_names, square=True
)
# Draw kept values over top
sns.heatmap(
    cos_sim_masked, ax=ax_b1, mask=mask_matrix, cmap=white_to_copper, vmin=0.0, vmax=1.0,
    cbar_kws={'orientation': 'horizontal', 'pad': 0.12, 'label': 'Cosine Similarity |CosSim| (|r| ≤ 0.75)', 'shrink': 0.8},
    annot=True, fmt=".2f", annot_kws={"size": 8.0, "weight": "bold"},
    linewidths=1.0, linecolor="#CBD5E1", xticklabels=all_gene_names, yticklabels=all_gene_names, square=True
)
for text in ax_b1.texts:
    if text.get_text():
        val = float(text.get_text())
        text.set_color("#FFFFFF" if val > 0.65 else "#1E293B")

# Mark pruned rows with small grey 'X'
for idx in dropped_indices:
    ax_b1.text(idx + 0.5, idx + 0.5, "DROP", ha="center", va="center", color="#94A3B8", fontsize=8, fontweight="bold")

ax_b1.tick_params(axis="x", rotation=45, labelsize=9.5, colors="#334155")
ax_b1.tick_params(axis="y", rotation=0, labelsize=9.5, colors="#334155")
plt.savefig(OUTPUT_FIG3B_OPT1, bbox_inches="tight", facecolor="#FFFFFF", edgecolor="none")
plt.close()

print("Generated test option 1 (masked 8x8)")
