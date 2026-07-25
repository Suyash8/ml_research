#!/usr/bin/env python3
"""
===============================================================================
STEP 07: MASTER AGGREGATION & FINAL CLINICAL DOSSIER GENERATION
===============================================================================
Input:  results/step_by_step_run/step06_patient_explainability.pkl
Output: results/step_by_step_run/step07_final_dossier.csv
        results/step_by_step_run/step07_final_dossier.json
===============================================================================
"""

import json
import pickle
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def main():
    print("\n" + "="*80)
    print("📋 STEP 07: MASTER AGGREGATION & FINAL CLINICAL DOSSIER GENERATION")
    print("="*80)

    input_pkl = ROOT_DIR / "results" / "step_by_step_run" / "step06_patient_explainability.pkl"
    output_dir = ROOT_DIR / "results" / "step_by_step_run"

    print(f"📥 Reading Intermediate Pipeline State: {input_pkl}")
    with open(input_pkl, "rb") as f:
        state = pickle.load(f)

    patient_ids = state["patient_ids"]
    risk_scores = state["risk_scores"]
    calibrated_results = state["calibrated_results"]
    mc_results = state["mc_results"]
    patient_explanations = state["patient_explanations"]

    rows = []
    for pid, eta, cal, mc, exp in zip(patient_ids, risk_scores, calibrated_results, mc_results, patient_explanations):
        row = {
            "PATIENT_ID": pid,
            "risk_score_eta": float(eta),
            "prob_survive_12m": cal.get("prob_survive_12m"),
            "prob_survive_24m": cal.get("prob_survive_24m"),
            "prob_survive_36m": cal.get("prob_survive_36m"),
            "prob_survive_60m": cal.get("prob_survive_60m"),
            "mc_p10_months": mc.get("mc_p10_months"),
            "mc_p50_median_months": mc.get("mc_p50_median_months"),
            "mc_p90_months": mc.get("mc_p90_months"),
            "mc_rmst_months": mc.get("mc_rmst_months"),
            "top_risk_drivers": ", ".join(d["feature"] for d in exp["top_risk_drivers"][:3]),
            "top_protective_drivers": ", ".join(d["feature"] for d in exp["top_protective_drivers"][:3]),
        }
        rows.append(row)

    df_dossier = pd.DataFrame(rows)

    # Save CSV
    csv_out = output_dir / "step07_final_dossier.csv"
    df_dossier.to_csv(csv_out, index=False)

    # Save JSON
    json_out = output_dir / "step07_final_dossier.json"
    full_report = {
        "pipeline_summary": {
            "n_patients_evaluated": len(patient_ids),
            "steps_completed": [
                "01_ingest_patient_data",
                "02_feature_transformation",
                "03_cox_risk_scoring",
                "04_isotonic_calibration",
                "05_monte_carlo_simulation",
                "06_xai_explainability",
                "07_generate_final_dossier"
            ]
        },
        "patient_dossiers": rows
    }

    with open(json_out, "w") as f:
        json.dump(full_report, f, indent=2)

    print("\n===============================================================================")
    print("🏆 FINAL PIPELINE DELIVERABLE: CLINICAL PATIENT DOSSIER")
    print("===============================================================================")
    print(df_dossier.to_string(index=False))
    print("===============================================================================\n")

    print(f"✅ STEP 07 COMPLETE!")
    print(f"📄 Final CSV Dossier:  {csv_out}")
    print(f"📄 Final JSON Dossier: {json_out}\n")

if __name__ == "__main__":
    main()
