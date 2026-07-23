import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

from ml_research.src.utils.plotting import (
    plot_mape_scatter,
    plot_global_importance,
    plot_group_summary,
    plot_pca_heatmaps,
    plot_pca_genes_heatmap,
    plot_waterfall,
    plot_patient_heatmap
)

def main():
    parser = argparse.ArgumentParser(description="Generate plots for XAI outputs and predictions.")
    parser.add_argument("--predictions", type=Path, default=Path("results/main_predictions.csv"), help="Predictions CSV")
    parser.add_argument("--xai-dir", type=Path, default=Path("results/explainability"), help="Dir with XAI CSVs")
    parser.add_argument("--out-dir", type=Path, default=Path("results/plots"), help="Output Directory for plots")
    args = parser.parse_args()
    
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Plot MAPE
    if args.predictions.exists():
        df = pd.read_csv(args.predictions)
        df_test = df[(df["split"].astype(str).str.lower() == "test") & (df["OS_EVENT"] == 1)].copy()
        
        if not df_test.empty:
            actual = df_test["OS_MONTHS"]
            predicted = df_test["mc_survival_p50_months"]
            
            valid_mask = actual.notna() & predicted.notna() & (actual > 0)
            actual = actual[valid_mask]
            predicted = predicted[valid_mask]
            
            if len(actual) > 0:
                mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                print(f"Generating MAPE Plot (MAPE = {mape:.2f}%)...")
                plot_mape_scatter(actual, predicted, mape, out_dir)
                
                with open(out_dir / "mape_metrics.json", "w") as f:
                    json.dump({"mape_percentage": mape, "n_patients_evaluated": len(actual)}, f, indent=4)

    # 2. Plot XAI
    in_dir = args.xai_dir
    if in_dir.exists():
        try:
            global_df = pd.read_csv(in_dir / "xai_global_feature_importance.csv")
            group_df = pd.read_csv(in_dir / "xai_group_summary.csv")
            pca_df = pd.read_csv(in_dir / "xai_pca_component_loadings.csv")
            detail_df = pd.read_csv(in_dir / "xai_patient_feature_contributions_test.csv")
            summary_df = pd.read_csv(in_dir / "xai_patient_summary_test.csv")
            
            global_df = global_df[~global_df["feature_name"].astype(str).str.contains("unknown", case=False, na=False)]
            detail_df = detail_df[~detail_df["feature_name"].astype(str).str.contains("unknown", case=False, na=False)]
            
            print("Generating Global Importance Plot...")
            plot_global_importance(global_df, out_dir)
            
            print("Generating Group Summary Plot...")
            plot_group_summary(group_df, out_dir)
            
            print("Generating PCA Heatmaps...")
            plot_pca_heatmaps(pca_df, out_dir)
            
            print("Generating PCA Genes Heatmap...")
            plot_pca_genes_heatmap(pca_df, out_dir)
            
            print("Generating Patient Heatmap...")
            plot_patient_heatmap(detail_df, out_dir)
            
            print("Generating Patient Waterfall Plots...")
            summary_df_sorted = summary_df.sort_values("recomputed_log_risk")
            if len(summary_df_sorted) >= 5:
                indices = np.linspace(0, len(summary_df_sorted) - 1, 5).astype(int)
                for i, idx in enumerate(indices):
                    pat_id = summary_df_sorted.iloc[idx]["PATIENT_ID"]
                    plot_waterfall(pat_id, detail_df, summary_df, out_dir)
                    print(f"Plotted patient {i+1}/5 (risk rank {idx}): {pat_id}")
            elif len(summary_df_sorted) > 0:
                for pat_id in summary_df_sorted["PATIENT_ID"]:
                    plot_waterfall(pat_id, detail_df, summary_df, out_dir)
                    
        except Exception as e:
            print(f"Error loading or plotting XAI CSVs: {e}")
            
    print(f"All plots saved to {out_dir.resolve()}")

if __name__ == "__main__":
    main()
