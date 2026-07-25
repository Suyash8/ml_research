#!/usr/bin/env python3
"""
===============================================================================
STEP 03: COX ELASTIC-NET LOG-HAZARD RISK SCORING
===============================================================================
Input:  results/step_by_step_run/step02_transformed_features.pkl
        model_weights/final_locked_model.pkl
Output: results/step_by_step_run/step03_risk_scores.json
        results/step_by_step_run/step03_risk_scores.pkl
===============================================================================
"""

import json
import pickle
import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def main():
    print("\n" + "="*80)
    print("⚙️ STEP 03: COX ELASTIC-NET LOG-HAZARD RISK SCORING")
    print("="*80)

    input_pkl = ROOT_DIR / "results" / "step_by_step_run" / "step02_transformed_features.pkl"
    model_pkl = ROOT_DIR / "model_weights" / "final_locked_model.pkl"
    output_dir = ROOT_DIR / "results" / "step_by_step_run"

    print(f"📥 Loading Step 02 Transformed Matrix: {input_pkl}")
    with open(input_pkl, "rb") as f:
        data_step02 = pickle.load(f)

    X_transformed = data_step02["X_transformed"]
    patient_ids = data_step02["patient_ids"]

    # Register CoxElasticNet for unpickling
    from src.models.cox_enet import CoxElasticNet
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "CoxElasticNet"):
        setattr(main_module, "CoxElasticNet", CoxElasticNet)

    with open(model_pkl, "rb") as f:
        artifact = pickle.load(f)

    cox_model = artifact["cox_model"]
    coefs = np.asarray(cox_model.coef_, dtype=float)

    print(f"🧮 Model Weights Loaded: {len(coefs)} Coefficients (Elastic-Net Regularized)")
    print("⚙️ Computing Linear Risk Scores (eta_i = X_i @ beta)...")

    risk_scores = np.asarray(cox_model.predict_risk(X_transformed), dtype=float)

    patient_risk_map = []
    for pid, eta in zip(patient_ids, risk_scores):
        tier = "High Risk" if eta > 0 else "Low Risk"
        patient_risk_map.append({
            "PATIENT_ID": pid,
            "risk_score_eta": float(eta),
            "risk_tier": tier
        })
        print(f"   • {pid}: Risk Score (eta) = {eta:+.4f} [{tier}]")

    # Save PKL
    pkl_out = output_dir / "step03_risk_scores.pkl"
    out_data = {
        "df_input": data_step02["df_input"],
        "patient_ids": patient_ids,
        "risk_scores": risk_scores,
        "X_transformed": X_transformed,
        "feature_names": data_step02["feature_names"],
        "clinical_names": data_step02["clinical_names"],
        "expr_cols": data_step02["expr_cols"]
    }
    with open(pkl_out, "wb") as f:
        pickle.dump(out_data, f)

    # Save JSON summary
    json_out = output_dir / "step03_risk_scores.json"
    json_summary = {
        "step": "03_cox_risk_scoring",
        "n_patients": len(patient_ids),
        "formula": "eta_i = X_i @ beta",
        "patient_risk_scores": patient_risk_map,
        "output_file": str(json_out)
    }
    with open(json_out, "w") as f:
        json.dump(json_summary, f, indent=2)

    print(f"✅ STEP 03 COMPLETE!")
    print(f"💾 Saved Intermediate Output: {pkl_out}")
    print(f"💾 Saved JSON Summary:        {json_out}\n")

if __name__ == "__main__":
    main()
