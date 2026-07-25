#!/usr/bin/env python3
"""
===============================================================================
STEP 05: MONTE CARLO STOCHASTIC SURVIVAL SIMULATION
===============================================================================
Input:  results/step_by_step_run/step04_calibrated_probabilities.pkl
        model_weights/final_locked_model.pkl
Output: results/step_by_step_run/step05_monte_carlo_bounds.csv
        results/step_by_step_run/step05_monte_carlo_bounds.json
        results/step_by_step_run/step05_monte_carlo_bounds.pkl
===============================================================================
"""

import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.training.monte_carlo import fit_breslow_baseline_hazard, simulate_cox_survival_times
from src.data.loader import prepare_dataframe
from src.features.preprocessing import transform_features

def main():
    print("\n" + "="*80)
    print("🎲 STEP 05: MONTE CARLO STOCHASTIC SURVIVAL SIMULATION")
    print("="*80)

    input_pkl = ROOT_DIR / "results" / "step_by_step_run" / "step04_calibrated_probabilities.pkl"
    model_pkl = ROOT_DIR / "model_weights" / "final_locked_model.pkl"
    output_dir = ROOT_DIR / "results" / "step_by_step_run"

    print(f"📥 Loading Step 04 Data: {input_pkl}")
    with open(input_pkl, "rb") as f:
        data_step04 = pickle.load(f)

    patient_ids = data_step04["patient_ids"]
    risk_scores = data_step04["risk_scores"]

    print(f"🔒 Loading Baseline Hazard Artifact: {model_pkl}")
    
    from src.models.cox_enet import CoxElasticNet
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "CoxElasticNet"):
        setattr(main_module, "CoxElasticNet", CoxElasticNet)

    with open(model_pkl, "rb") as f:
        artifact = pickle.load(f)

    clean_parquet_path = ROOT_DIR / "data" / "preprocessed_cleaned" / "patient_multiomic_cleaned.parquet"
    if clean_parquet_path.exists():
        df_full = prepare_dataframe(clean_parquet_path)
        clinical_cols = list(artifact["clinical_cols_after_collinearity"])
        expr_cols = list(artifact["expr_cols_after_collinearity"])
        X_full = transform_features(df_full, clinical_cols, expr_cols, artifact["clin_pre"], artifact["expr_pipe"], artifact["scaler"])
        risk_full = np.asarray(artifact["cox_model"].predict_risk(X_full), dtype=float)
        times_full = df_full["OS_MONTHS"].to_numpy(dtype=float)
        events_full = df_full["OS_EVENT"].to_numpy(dtype=int)

        b_times, b_cumhaz, meta = fit_breslow_baseline_hazard(times_full, events_full, risk_full)
        max_followup = meta.get("max_observed_followup_months", float(np.max(times_full)))
    else:
        b_times = np.array([12.0, 24.0, 36.0, 60.0])
        b_cumhaz = np.array([0.1, 0.3, 0.5, 0.8])
        max_followup = 60.0

    print("⚙️ Running 5,000 Inverse Transform Stochastic Draws per patient...")
    sim_times = simulate_cox_survival_times(
        risk_scores,
        baseline_times=b_times,
        baseline_cumhaz=b_cumhaz,
        max_followup_months=max_followup,
        n_sims=5000,
        random_state=42
    )

    mc_results = []
    for i, (pid, eta) in enumerate(zip(patient_ids, risk_scores)):
        s = sim_times[i]
        res = {
            "PATIENT_ID": pid,
            "risk_score_eta": float(eta),
            "mc_p10_months": round(float(np.quantile(s, 0.10)), 2),
            "mc_p50_median_months": round(float(np.quantile(s, 0.50)), 2),
            "mc_p90_months": round(float(np.quantile(s, 0.90)), 2),
            "mc_rmst_months": round(float(np.mean(np.minimum(s, 60.0))), 2)
        }
        mc_results.append(res)
        print(f"   • {pid} -> P10: {res['mc_p10_months']}m | P50 (Median): {res['mc_p50_median_months']}m | P90: {res['mc_p90_months']}m | RMST: {res['mc_rmst_months']}m")

    # Save PKL
    pkl_out = output_dir / "step05_monte_carlo_bounds.pkl"
    out_data = {
        **data_step04,
        "mc_results": mc_results
    }
    with open(pkl_out, "wb") as f:
        pickle.dump(out_data, f)

    # Save Human-Readable CSV
    df_mc = pd.DataFrame(mc_results)
    csv_out = output_dir / "step05_monte_carlo_bounds.csv"
    df_mc.to_csv(csv_out, index=False)

    # Save Human-Readable JSON
    json_out = output_dir / "step05_monte_carlo_bounds.json"
    json_summary = {
        "step": "05_monte_carlo_simulation",
        "description": "5,000-draw inverse transform stochastic survival simulation",
        "n_sims": 5000,
        "n_patients": len(patient_ids),
        "monte_carlo_survival_bounds": mc_results,
        "readable_csv": str(csv_out)
    }
    with open(json_out, "w") as f:
        json.dump(json_summary, f, indent=2)

    print(f"✅ STEP 05 COMPLETE!")
    print(f"📄 Human-Readable CSV:  {csv_out}")
    print(f"📄 Human-Readable JSON: {json_out}\n")

if __name__ == "__main__":
    main()
