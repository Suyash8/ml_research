import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Generate MAPE curve and scatter plot for v5 predictions.")
    parser.add_argument("--predictions", type=Path, default=Path("/home/illionar/Projects/ml_research/data/model_outputs/cox_enet_calibrated_mc_outputs_v5/main_predictions.csv"), help="Predictions CSV")
    parser.add_argument("--out-dir", type=Path, default=Path("/home/illionar/Projects/ml_research/data/model_outputs/cox_enet_calibrated_mc_outputs_v5_explainability"), help="Output Directory")
    args = parser.parse_args()
    
    df = pd.read_csv(args.predictions)
    
    # Filter to test set and only patients who experienced the event
    df_test = df[(df["split"].astype(str).str.lower() == "test") & (df["OS_EVENT"] == 1)].copy()
    
    if df_test.empty:
        print("No uncensored patients in the test set to compute MAPE.")
        return
        
    actual = df_test["OS_MONTHS"]
    predicted = df_test["mc_survival_p50_months"]
    
    # Drop NaNs and zeros to avoid division by zero
    valid_mask = actual.notna() & predicted.notna() & (actual > 0)
    actual = actual[valid_mask]
    predicted = predicted[valid_mask]
    
    if len(actual) == 0:
        print("No valid actual/predicted values.")
        return
        
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100  # as percentage
    
    # Plotting Scatter of Actual vs Predicted
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predicted, alpha=0.7, color="#1f77b4")
    
    # Plot identity line
    max_val = max(actual.max(), predicted.max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='Ideal (Actual = Predicted)')
    
    plt.title(f"Predicted vs Actual Survival Time (Test Set)\nMAPE: {mape:.2f}%")
    plt.xlabel("Actual Survival Time (Months)")
    plt.ylabel("Predicted Median Survival Time (Months)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "plot_mape.png", dpi=300)
    plt.close()
    
    # Save the results to JSON
    results = {
        "mape_percentage": mape,
        "n_patients_evaluated": len(actual)
    }
    with open(out_dir / "mape_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"MAPE = {mape:.2f}%")
    print(f"MAPE plot saved to {out_dir / 'plot_mape.png'}")

if __name__ == "__main__":
    main()
