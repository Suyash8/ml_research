import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

def plot_mape_scatter(actual: pd.Series, predicted: pd.Series, mape: float, out_dir: Path):
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predicted, alpha=0.7, color="#1f77b4")
    
    max_val = max(actual.max(), predicted.max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='Ideal (Actual = Predicted)')
    
    plt.title(f"Predicted vs Actual Survival Time (Test Set)\nMAPE: {mape:.2f}%")
    plt.xlabel("Actual Survival Time (Months)")
    plt.ylabel("Predicted Median Survival Time (Months)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_mape.png", dpi=300)
    plt.close()

def plot_global_importance(df: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(10, 8))
    top_df = df.sort_values("abs_coefficient", ascending=False).head(20).copy()
    top_df = top_df.sort_values("coefficient", ascending=True)
    colors = ["#d62728" if val > 0 else "#1f77b4" for val in top_df["coefficient"]]
    plt.barh(top_df["feature_name"], top_df["coefficient"], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Top 20 Global Feature Importances (Cox Coefficients)")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_global_importance.png", dpi=300)
    plt.close()

def plot_group_summary(df: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(6, 6))
    plt.pie(
        df["sum_abs_coefficient"], 
        labels=df["group"], 
        autopct='%1.1f%%',
        startangle=90,
        colors=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]
    )
    plt.title("Total Absolute Impact by Feature Group")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_group_summary.png", dpi=300)
    plt.close()

def plot_pca_heatmaps(pca_df: pd.DataFrame, out_dir: Path, top_n_pcs: int = 4):
    pc_importance = pca_df[["pc_name", "pc_coefficient"]].drop_duplicates().copy()
    pc_importance["abs_coef"] = pc_importance["pc_coefficient"].abs()
    top_pcs = pc_importance.sort_values("abs_coef", ascending=False)["pc_name"].head(top_n_pcs).tolist()
    
    if not top_pcs:
        return

    fig, axes = plt.subplots(1, len(top_pcs), figsize=(4 * len(top_pcs), 6), sharey=False)
    if len(top_pcs) == 1:
        axes = [axes]
        
    for ax, pc in zip(axes, top_pcs):
        pc_data = pca_df[pca_df["pc_name"] == pc].sort_values("rank_within_pc")
        if pc_data.empty:
            continue
            
        genes = pc_data["gene_name"].values
        loadings = pc_data["risk_weighted_loading"].values
        colors = ["#d62728" if val > 0 else "#1f77b4" for val in loadings]
        
        y_pos = np.arange(len(genes))
        ax.barh(y_pos, loadings, align='center', color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(genes)
        ax.invert_yaxis()
        ax.set_title(f"{pc}\n(coef: {pc_data['pc_coefficient'].iloc[0]:.3f})")
        ax.set_xlabel("Risk Weighted Loading")
        ax.axvline(0, color="black", linewidth=0.5)

    plt.suptitle("Top Genes Driving the Most Important Expression PCs", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_pca_top_genes.png", dpi=300)
    plt.close()

def plot_pca_genes_heatmap(pca_df: pd.DataFrame, out_dir: Path):
    pivot_df = pca_df.pivot_table(index="gene_name", columns="pc_name", values="risk_weighted_loading", fill_value=0.0)
    if pivot_df.empty:
        return
        
    max_abs = pivot_df.abs().max(axis=1).sort_values(ascending=False)
    pivot_df = pivot_df.loc[max_abs.index]
    
    plt.figure(figsize=(10, max(6, len(pivot_df) * 0.2)))
    sns.heatmap(pivot_df, cmap="RdBu_r", center=0, cbar_kws={'label': 'Risk Weighted Loading'})
    plt.title("Important Genes across PCs (Risk Weighted Loadings)")
    plt.xlabel("Principal Component")
    plt.ylabel("Gene Marker")
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
        
    p_detail = p_detail.sort_values("contribution", ascending=True)
    contributions = p_detail["contribution"].values
    features = p_detail["feature_name"].values
    
    plt.figure(figsize=(10, 14))
    colors = ["#d62728" if val > 0 else "#1f77b4" for val in contributions]
    plt.barh(features, contributions, color=colors)
    plt.axvline(0, color="black", linewidth=1)
    
    total_risk = p_summary["recomputed_log_risk"].iloc[0]
    os_months = p_summary["OS_MONTHS"].iloc[0]
    os_event = p_summary["OS_EVENT"].iloc[0]
    event_str = "Deceased" if os_event == 1 else "Censored"
    
    plt.title(f"Patient {patient_id} Risk Contributions\nTotal Log-Risk: {total_risk:.3f} | Survival: {os_months:.1f} mo ({event_str})")
    plt.xlabel("Contribution to Log-Risk (Cox Scale)")
    plt.tight_layout()
    plt.savefig(out_dir / f"plot_waterfall_patient_{patient_id}.png", dpi=300)
    plt.close()

def plot_patient_heatmap(detail_df: pd.DataFrame, out_dir: Path):
    pivot_df = detail_df.pivot_table(index="PATIENT_ID", columns="feature_name", values="contribution", fill_value=0.0)
    pivot_df = pivot_df.dropna(how='all', axis=0).dropna(how='all', axis=1)
    
    mean_abs_contrib = pivot_df.abs().mean().sort_values(ascending=False)
    pivot_df = pivot_df[mean_abs_contrib.index]
    
    fig, ax = plt.subplots(figsize=(20, 16))
    c = ax.imshow(pivot_df.values, cmap="RdBu_r", aspect="auto")
    vmax = np.nanmax(np.abs(pivot_df.values))
    c.set_clim(-vmax, vmax)
    plt.colorbar(c, ax=ax, label="Contribution to Log-Risk")
    
    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns, rotation=90, fontsize=8)
    if len(pivot_df.index) <= 100:
        ax.set_yticks(np.arange(len(pivot_df.index)))
        ax.set_yticklabels(pivot_df.index, fontsize=8)
    else:
        ax.set_yticks([])
        
    ax.set_title("Heatmap of Feature Contributions per Patient")
    ax.set_xlabel("Features")
    ax.set_ylabel("Patients")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_patient_heatmap.png", dpi=300)
    plt.close()
