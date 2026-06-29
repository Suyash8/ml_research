import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc

def compute_pr_auc_for_horizon(df, horizon_months, prob_col):
    """
    Computes Precision-Recall AUC for a specific horizon.
    Returns (precision, recall, pr_auc_value).
    """
    # 1. Event occurred on or before horizon
    is_event = (df["OS_EVENT"] == 1) & (df["OS_MONTHS"] <= horizon_months)
    # 2. Patient survived past horizon (can be censored later or event later)
    survived = df["OS_MONTHS"] > horizon_months
    
    # We drop patients censored before horizon
    valid_mask = is_event | survived
    df_valid = df[valid_mask].copy()
    
    if df_valid.empty:
        return None, None, float('nan')
        
    y_true = is_event[valid_mask].astype(int)
    y_scores = df_valid[prob_col]
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    return precision, recall, pr_auc

def main():
    parser = argparse.ArgumentParser(description="Generate PR AUC curves for v5 predictions.")
    parser.add_argument("--predictions", type=Path, default=Path("/home/illionar/Projects/ml_research/data/model_outputs/cox_enet_calibrated_mc_outputs_v5/main_predictions.csv"), help="Predictions CSV")
    parser.add_argument("--out-dir", type=Path, default=Path("/home/illionar/Projects/ml_research/data/model_outputs/cox_enet_calibrated_mc_outputs_v5_explainability"), help="Output Directory")
    args = parser.parse_args()
    
    df = pd.read_csv(args.predictions)
    # Filter to test set
    df_test = df[df["split"].astype(str).str.lower() == "test"].copy()
    
    horizons = [12, 24, 36, 60]
    
    plt.figure(figsize=(8, 6))
    
    pr_auc_results = {}
    
    for h in horizons:
        prob_col = f"cal_event_prob_{h}m"
        if prob_col not in df_test.columns:
            # Maybe test_cox_survival_prob_12m is there
            prob_col_survival = f"test_cox_survival_prob_{h}m"
            if prob_col_survival in df_test.columns:
                df_test[prob_col] = 1.0 - df_test[prob_col_survival]
            else:
                continue
                
        precision, recall, pr_auc = compute_pr_auc_for_horizon(df_test, h, prob_col)
        if precision is not None:
            plt.plot(recall, precision, label=f"{h}-month (AUC = {pr_auc:.3f})")
            pr_auc_results[f"pr_auc_{h}m"] = pr_auc
            print(f"Horizon {h} months: PR AUC = {pr_auc:.3f}")
            
    plt.title("Precision-Recall Curves (Test Set)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "plot_pr_auc.png", dpi=300)
    plt.close()
    
    # Save the results to JSON
    with open(out_dir / "pr_auc_metrics.json", "w") as f:
        json.dump(pr_auc_results, f, indent=4)
        
    print(f"PR AUC plot saved to {out_dir / 'plot_pr_auc.png'}")

if __name__ == "__main__":
    main()
