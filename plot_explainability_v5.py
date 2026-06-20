import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_global_importance(df, out_dir):
    """Plot the top risk-increasing and risk-decreasing features overall."""
    plt.figure(figsize=(10, 8))
    
    # Sort by absolute coefficient for top features, take top 20
    top_df = df.sort_values("abs_coefficient", ascending=False).head(20).copy()
    
    # Sort by actual coefficient for plotting
    top_df = top_df.sort_values("coefficient", ascending=True)
    
    colors = ["#d62728" if val > 0 else "#1f77b4" for val in top_df["coefficient"]]
    
    bars = plt.barh(top_df["feature_name"], top_df["coefficient"], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    
    plt.title("Top 20 Global Feature Importances (Cox Coefficients)")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_global_importance.png", dpi=300)
    plt.close()


def plot_group_summary(df, out_dir):
    """Plot clinical vs expression total impact."""
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


def plot_pca_heatmaps(pca_df, out_dir, top_n_pcs=4):
    """Plot heatmaps for the top genes driving the most important PCA components."""
    # Find the top PCs based on absolute pc_coefficient
    pc_importance = pca_df[["pc_name", "pc_coefficient"]].drop_duplicates()
    pc_importance["abs_coef"] = pc_importance["pc_coefficient"].abs()
    top_pcs = pc_importance.sort_values("abs_coef", ascending=False)["pc_name"].head(top_n_pcs).tolist()
    
    fig, axes = plt.subplots(1, len(top_pcs), figsize=(4 * len(top_pcs), 6), sharey=False)
    if len(top_pcs) == 1:
        axes = [axes]
        
    for ax, pc in zip(axes, top_pcs):
        pc_data = pca_df[pca_df["pc_name"] == pc].sort_values("rank_within_pc")
        # Ensure we don't try to plot if no data
        if pc_data.empty:
            continue
            
        genes = pc_data["gene_name"].values
        loadings = pc_data["risk_weighted_loading"].values
        
        colors = ["#d62728" if val > 0 else "#1f77b4" for val in loadings]
        
        y_pos = np.arange(len(genes))
        ax.barh(y_pos, loadings, align='center', color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(genes)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_title(f"{pc}\n(coef: {pc_data['pc_coefficient'].iloc[0]:.3f})")
        ax.set_xlabel("Risk Weighted Loading")
        ax.axvline(0, color="black", linewidth=0.5)

    plt.suptitle("Top Genes Driving the Most Important Expression PCs", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_pca_top_genes.png", dpi=300)
    plt.close()


def plot_waterfall(patient_id, detail_df, summary_df, out_dir):
    """Plot a waterfall chart for a specific patient."""
    p_detail = detail_df[detail_df["PATIENT_ID"] == str(patient_id)].copy()
    if p_detail.empty:
        print(f"No detail data found for patient {patient_id}")
        return
        
    p_summary = summary_df[summary_df["PATIENT_ID"] == str(patient_id)]
    if p_summary.empty:
        print(f"No summary data found for patient {patient_id}")
        return
        
    # Sort by contribution so we can see all features across the spectrum
    p_detail = p_detail.sort_values("contribution", ascending=True)
    
    contributions = p_detail["contribution"].values
    features = p_detail["feature_name"].values
    
    plt.figure(figsize=(10, 14))
    
    # We will use a bidirectional bar chart (SHAP-style local explanation).
    
    colors = ["#d62728" if val > 0 else "#1f77b4" for val in contributions]
    bars = plt.barh(features, contributions, color=colors)
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


def plot_patient_heatmap(detail_df, out_dir):
    """Plot a heatmap of all patient vs all feature contributions."""
    pivot_df = detail_df.pivot(index="PATIENT_ID", columns="feature_name", values="contribution")
    pivot_df = pivot_df.dropna(how='all', axis=0).dropna(how='all', axis=1)
    
    # Sort columns by mean absolute contribution
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


def main():
    parser = argparse.ArgumentParser(description="Generate plots for XAI outputs.")
    parser.add_argument("--input-dir", type=Path, default=Path("/home/illionar/Projects/ml_research/data/model_outputs/cox_enet_calibrated_mc_outputs_v5_explainability"), help="Dir with CSVs")
    args = parser.parse_args()
    
    in_dir = args.input_dir
    if not in_dir.exists():
        print(f"Error: Input directory {in_dir} does not exist.")
        return
        
    # Load data
    try:
        global_df = pd.read_csv(in_dir / "xai_global_feature_importance.csv")
        group_df = pd.read_csv(in_dir / "xai_group_summary.csv")
        pca_df = pd.read_csv(in_dir / "xai_pca_component_loadings.csv")
        detail_df = pd.read_csv(in_dir / "xai_patient_feature_contributions_test.csv")
        summary_df = pd.read_csv(in_dir / "xai_patient_summary_test.csv")
        
        # Filter out features that are 'unknown'
        global_df = global_df[~global_df["feature_name"].astype(str).str.contains("unknown", case=False, na=False)]
        detail_df = detail_df[~detail_df["feature_name"].astype(str).str.contains("unknown", case=False, na=False)]
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return
        
    print("Generating Global Importance Plot...")
    plot_global_importance(global_df, in_dir)
    
    print("Generating Group Summary Plot...")
    plot_group_summary(group_df, in_dir)
    
    print("Generating PCA Heatmaps...")
    plot_pca_heatmaps(pca_df, in_dir)
    
    print("Generating Patient Heatmap...")
    plot_patient_heatmap(detail_df, in_dir)
    
    print("Generating Patient Waterfall Plots...")
    # Pick a high risk patient and a low risk patient
    summary_df_sorted = summary_df.sort_values("recomputed_log_risk")
    if len(summary_df_sorted) > 0:
        low_risk_pat = summary_df_sorted.iloc[0]["PATIENT_ID"]
        plot_waterfall(low_risk_pat, detail_df, summary_df, in_dir)
        print(f"Plotted low risk patient: {low_risk_pat}")
        
        high_risk_pat = summary_df_sorted.iloc[-1]["PATIENT_ID"]
        plot_waterfall(high_risk_pat, detail_df, summary_df, in_dir)
        print(f"Plotted high risk patient: {high_risk_pat}")

    print(f"All plots saved to {in_dir.resolve()}")

if __name__ == "__main__":
    main()
