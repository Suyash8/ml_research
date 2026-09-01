import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

def clean_feature_name(name: str) -> str:
    """Clean feature names by removing noisy prefixes like cat__, num__, EXPR_."""
    name = str(name)
    if name.startswith("cat__"):
        name = name[5:]
    if name.startswith("num__"):
        name = name[5:]
    if name.startswith("EXPR_"):
        name = name[5:]
    return name

def plot_mape_scatter(actual: pd.Series, predicted: pd.Series, mape: float, out_dir: Path):
    plt.figure(figsize=(6, 5))
    plt.scatter(actual, predicted, alpha=0.7, color="#1f77b4")
    
    max_val = max(actual.max(), predicted.max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='Ideal (Actual = Predicted)')
    
    plt.title(f"Predicted vs Actual Survival Time (Test Set)\nMAPE: {mape:.2f}%", fontsize=11, fontweight="bold")
    plt.xlabel("Actual Survival Time (Months)", fontsize=10)
    plt.ylabel("Predicted Median Survival Time (Months)", fontsize=10)
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_mape.png", dpi=300)
    plt.close()

def plot_global_importance(df: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(6.5, 3.4))
    top_df = df.sort_values("abs_coefficient", ascending=False).head(10).copy()
    top_df["feature_clean"] = top_df["feature_name"].apply(clean_feature_name)
    top_df = top_df.sort_values("coefficient", ascending=True)
    colors = ["#d62728" if val > 0 else "#1f77b4" for val in top_df["coefficient"]]
    
    bars = plt.barh(top_df["feature_clean"], top_df["coefficient"], color=colors, height=0.6)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Top 10 Global Feature Importances", fontsize=11, fontweight="bold")
    plt.xlabel("Cox Coefficient Value", fontsize=9.5)
    plt.ylabel("Feature", fontsize=9.5)
    plt.tick_params(axis='both', labelsize=8.5)
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_global_importance.png", dpi=300)
    plt.close()

def plot_group_summary(df: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(5, 5))
    plt.pie(
        df["sum_abs_coefficient"], 
        labels=df["group"], 
        autopct='%1.1f%%',
        startangle=90,
        colors=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]
    )
    plt.title("Total Absolute Impact by Feature Group", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_group_summary.png", dpi=300)
    plt.close()

def plot_pca_heatmaps(pca_df: pd.DataFrame, out_dir: Path, top_n_pcs: int = 4):
    pc_importance = pca_df[["pc_name", "pc_coefficient"]].drop_duplicates().copy()
    pc_importance["abs_coef"] = pc_importance["pc_coefficient"].abs()
    top_pcs = pc_importance.sort_values("abs_coef", ascending=False)["pc_name"].head(top_n_pcs).tolist()
    
    if not top_pcs:
        return

    fig, axes = plt.subplots(1, len(top_pcs), figsize=(3.2 * len(top_pcs), 4.5), sharey=False)
    if len(top_pcs) == 1:
        axes = [axes]
        
    for ax, pc in zip(axes, top_pcs):
        pc_data = pca_df[pca_df["pc_name"] == pc].sort_values("rank_within_pc")
        if pc_data.empty:
            continue
            
        genes = [clean_feature_name(g) for g in pc_data["gene_name"].values]
        loadings = pc_data["risk_weighted_loading"].values
        colors = ["#d62728" if val > 0 else "#1f77b4" for val in loadings]
        
        y_pos = np.arange(len(genes))
        ax.barh(y_pos, loadings, align='center', color=colors, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(genes, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{clean_feature_name(pc)}\n(coef: {pc_data['pc_coefficient'].iloc[0]:.3f})", fontsize=10)
        ax.set_xlabel("Risk Weighted Loading", fontsize=9)
        ax.axvline(0, color="black", linewidth=0.5)

    plt.suptitle("Top Genes Driving Important Expression PCs", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_pca_top_genes.png", dpi=300)
    plt.close()

def plot_pca_genes_heatmap(pca_df: pd.DataFrame, out_dir: Path):
    pca_df_clean = pca_df.copy()
    pca_df_clean["gene_name"] = pca_df_clean["gene_name"].apply(clean_feature_name)
    pca_df_clean["pc_name"] = pca_df_clean["pc_name"].apply(clean_feature_name)
    pivot_df = pca_df_clean.pivot_table(index="gene_name", columns="pc_name", values="risk_weighted_loading", fill_value=0.0)
    if pivot_df.empty:
        return
        
    max_abs = pivot_df.abs().max(axis=1).sort_values(ascending=False)
    pivot_df = pivot_df.loc[max_abs.index]
    
    plt.figure(figsize=(8, max(4.5, len(pivot_df) * 0.18)))
    sns.heatmap(pivot_df, cmap="RdBu_r", center=0, cbar_kws={'label': 'Risk Weighted Loading'})
    plt.title("Important Genes across PCs", fontsize=11, fontweight="bold")
    plt.xlabel("Principal Component", fontsize=9.5)
    plt.ylabel("Gene Marker", fontsize=9.5)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_pca_genes_heatmap.png", dpi=300)
    plt.close()

def plot_waterfall(patient_id: str, detail_df: pd.DataFrame, summary_df: pd.DataFrame, out_dir: Path):
    p_detail = detail_df[detail_df["PATIENT_ID"] == str(patient_id)].copy()
    if p_detail.empty:
        return
        
    p_summary = summary_df[summary_df["PATIENT_ID"] == str(patient_id)]
    if p_summary.empty:
        return
    
    # Filter to Top 10 features by absolute contribution magnitude for ultra-compact footprint
    p_detail["abs_contrib"] = p_detail["contribution"].abs()
    top10_detail = p_detail.sort_values("abs_contrib", ascending=False).head(10).copy()
    top10_detail["feature_clean"] = top10_detail["feature_name"].apply(clean_feature_name)
    top10_detail = top10_detail.sort_values("contribution", ascending=True)
    
    contributions = top10_detail["contribution"].values
    features = top10_detail["feature_clean"].values
    
    plt.figure(figsize=(6.5, 3.4))
    colors = ["#d62728" if val > 0 else "#1f77b4" for val in contributions]
    plt.barh(features, contributions, color=colors, height=0.6)
    plt.axvline(0, color="black", linewidth=0.8)
    
    total_risk = p_summary["recomputed_log_risk"].iloc[0]
    os_months = p_summary["OS_MONTHS"].iloc[0]
    os_event = p_summary["OS_EVENT"].iloc[0]
    event_str = "Deceased" if os_event == 1 else "Censored"
    
    plt.title(f"Patient {patient_id} Top 10 Risk Contributions\nRisk: {total_risk:.3f} | Survival: {os_months:.1f} mo ({event_str})", fontsize=10.5, fontweight="bold")
    plt.xlabel("Contribution to Log-Risk (Cox Scale)", fontsize=9.5)
    plt.ylabel("Feature", fontsize=9.5)
    plt.tick_params(axis='both', labelsize=8.5)
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / f"plot_waterfall_patient_{patient_id}.png", dpi=300)
    plt.close()

def plot_patient_heatmap(detail_df: pd.DataFrame, out_dir: Path):
    detail_clean = detail_df.copy()
    detail_clean["feature_name"] = detail_clean["feature_name"].apply(clean_feature_name)
    pivot_df = detail_clean.pivot_table(index="PATIENT_ID", columns="feature_name", values="contribution", fill_value=0.0)
    pivot_df = pivot_df.dropna(how='all', axis=0).dropna(how='all', axis=1)
    
    mean_abs_contrib = pivot_df.abs().mean().sort_values(ascending=False)
    pivot_df = pivot_df[mean_abs_contrib.index]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    c = ax.imshow(pivot_df.values, cmap="RdBu_r", aspect="auto")
    vmax = np.nanmax(np.abs(pivot_df.values))
    c.set_clim(-vmax, vmax)
    plt.colorbar(c, ax=ax, label="Contribution to Log-Risk")
    
    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns, rotation=90, fontsize=7.5)
    if len(pivot_df.index) <= 100:
        ax.set_yticks(np.arange(len(pivot_df.index)))
        ax.set_yticklabels(pivot_df.index, fontsize=7.5)
    else:
        ax.set_yticks([])
        
    ax.set_title("Heatmap of Feature Contributions per Patient", fontsize=11, fontweight="bold")
    ax.set_xlabel("Features", fontsize=9.5)
    ax.set_ylabel("Patients", fontsize=9.5)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_patient_heatmap.png", dpi=300)
    plt.close()
