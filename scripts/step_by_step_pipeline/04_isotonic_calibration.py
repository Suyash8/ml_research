#!/usr/bin/env python3
"""
===============================================================================
STEP 04: ISOTONIC SURVIVAL PROBABILITY CALIBRATION
===============================================================================
Input:  results/step_by_step_run/step03_risk_scores.pkl
        model_weights/final_locked_model.pkl
Output: results/step_by_step_run/step04_calibrated_probabilities.csv
        results/step_by_step_run/step04_calibrated_probabilities.json
        results/step_by_step_run/step04_calibrated_probabilities.pkl
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

def main():
    print("\n" + "="*80)
    print("📊 STEP 04: ISOTONIC SURVIVAL PROBABILITY CALIBRATION")
    print("="*80)

    input_pkl = ROOT_DIR / "results" / "step_by_step_run" / "step03_risk_scores.pkl"
    model_pkl = ROOT_DIR / "model_weights" / "final_locked_model.pkl"
    output_dir = ROOT_DIR / "results" / "step_by_step_run"

    print(f"📥 Loading Step 03 Risk Scores: {input_pkl}")
    with open(input_pkl, "rb") as f:
        data_step03 = pickle.load(f)

    patient_ids = data_step03["patient_ids"]
    risk_scores = data_step03["risk_scores"]

    print(f"🔒 Loading Isotonic Calibrators: {model_pkl}")
    
    # Register CoxElasticNet for unpickling
    from src.models.cox_enet import CoxElasticNet
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "CoxElasticNet"):
        setattr(main_module, "CoxElasticNet", CoxElasticNet)

    with open(model_pkl, "rb") as f:
        artifact = pickle.load(f)

    calibrators = artifact.get("horizon_calibrators", {})
    horizons = [12.0, 24.0, 36.0, 60.0]

    print("⚙️ Evaluating Non-Parametric PAVA Isotonic Models for 12, 24, 36, 60 months...")

    calibrated_results = []
    for i, (pid, eta) in enumerate(zip(patient_ids, risk_scores)):
        patient_probs = {"PATIENT_ID": pid, "risk_score_eta": float(eta)}
        for h in horizons:
            cal_info = calibrators.get(h)
            if isinstance(cal_info, dict) and "isotonic" in cal_info and cal_info["isotonic"] is not None:
                iso_model = cal_info["isotonic"]
                event_prob = float(iso_model.predict([eta])[0])
                surv_prob = float(np.clip(1.0 - event_prob, 0.0, 1.0))
                patient_probs[f"prob_survive_{int(h)}m"] = round(surv_prob, 4)
            else:
                patient_probs[f"prob_survive_{int(h)}m"] = None
        
        calibrated_results.append(patient_probs)
        print(f"   • {pid} [eta={eta:+.3f}] -> 12m: {patient_probs['prob_survive_12m']*100:.1f}% | 36m: {patient_probs['prob_survive_36m']*100:.1f}% | 60m: {patient_probs['prob_survive_60m']*100:.1f}%")

    # Save PKL
    pkl_out = output_dir / "step04_calibrated_probabilities.pkl"
    out_data = {
        **data_step03,
        "calibrated_results": calibrated_results
    }
    with open(pkl_out, "wb") as f:
        pickle.dump(out_data, f)

    # Save Human-Readable CSV
    df_cal = pd.DataFrame(calibrated_results)
    csv_out = output_dir / "step04_calibrated_probabilities.csv"
    df_cal.to_csv(csv_out, index=False)

    # Save Human-Readable JSON
    json_out = output_dir / "step04_calibrated_probabilities.json"
    json_summary = {
        "step": "04_isotonic_calibration",
        "description": "Monotonic mapping from relative risk score eta to absolute survival probabilities",
        "n_patients": len(patient_ids),
        "horizons_months": [12, 24, 36, 60],
        "calibrated_survival_probabilities": calibrated_results,
        "readable_csv": str(csv_out)
    }
    with open(json_out, "w") as f:
        json.dump(json_summary, f, indent=2)

    print(f"✅ STEP 04 COMPLETE!")
    print(f"📄 Human-Readable CSV:  {csv_out}")
    print(f"📄 Human-Readable JSON: {json_out}\n")

if __name__ == "__main__":
    main()
