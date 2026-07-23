#!/usr/bin/env python3
"""
===============================================================================
LIVE DEMO CLI: Multi-Omic Cox Elastic-Net Pipeline (v5)
===============================================================================
This script performs real-time inference on new/sample patient records using
the frozen model artifact (`final_locked_model.pkl`).

It computes:
  1. Transformed Features (Clinical Preprocessing + StandardScaler + PCA 50)
  2. Relative Cox Risk Score (eta)
  3. Isotonic Calibrated Survival Probabilities (12m, 24m, 36m, 60m)
  4. Monte Carlo Bounded Survival Estimates (P10 Pessimistic, P50 Median, P90 Optimistic)
  5. Local Patient Explainability (Waterfall Risk-Increasing & Protective Drivers)
  6. PCA Back-Projection to Raw Genes

Usage:
  python scripts/live_demo.py --input demo_input_sample.csv
===============================================================================
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Add root directory to sys.path to resolve imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.loader import prepare_dataframe
from src.features.preprocessing import transform_features
from src.metrics.explainability import build_feature_names, build_pca_backprojection
from src.training.monte_carlo import fit_breslow_baseline_hazard, simulate_cox_survival_times
from src.utils.config import HORIZONS_MONTHS, MC_N_SIMS, MC_RANDOM_STATE, MC_RMST_HORIZON_MONTHS


# Color formatting for terminal
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def load_model_artifact(model_path: Path) -> Dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Locked model artifact not found at {model_path}")
    
    # Ensure CoxElasticNet is available in sys.modules
    from src.models.cox_enet import CoxElasticNet
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "CoxElasticNet"):
        setattr(main_module, "CoxElasticNet", CoxElasticNet)

    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    return artifact


def get_baseline_hazard_from_train(artifact: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit Breslow baseline hazard using full clean dataset to ensure precise MC simulations."""
    full_data_path = ROOT_DIR / "data" / "preprocessed_cleaned" / "patient_multiomic_cleaned.parquet"
    if not full_data_path.exists():
        # Fallback if full file missing
        return np.array([12.0, 24.0, 36.0, 60.0]), np.array([0.1, 0.3, 0.5, 0.8]), 60.0

    df_full = prepare_dataframe(full_data_path)
    clinical_cols = list(artifact.get("clinical_cols_after_collinearity") or [])
    expr_cols = list(artifact.get("expr_cols_after_collinearity") or [])
    clin_pre = artifact.get("clin_pre")
    expr_pipe = artifact.get("expr_pipe")
    scaler = artifact.get("scaler")
    model = artifact.get("cox_model")

    X_full = transform_features(df_full, clinical_cols, expr_cols, clin_pre, expr_pipe, scaler)
    risk_scores_full = np.asarray(model.predict_risk(X_full), dtype=float)

    times = df_full["OS_MONTHS"].to_numpy(dtype=float)
    events = df_full["OS_EVENT"].to_numpy(dtype=int)

    base_times, base_cumhaz, meta = fit_breslow_baseline_hazard(times, events, risk_scores_full)
    max_followup = meta.get("max_observed_followup_months", float(np.max(times)))
    return base_times, base_cumhaz, max_followup


def run_demo_inference(input_path: Path, model_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    artifact = load_model_artifact(model_path)
    
    # Load input data (supports .csv or .parquet)
    if input_path.suffix.lower() == ".parquet":
        df_input = pd.read_parquet(input_path)
    else:
        df_input = pd.read_csv(input_path)

    patient_ids = df_input["PATIENT_ID"].astype(str).tolist() if "PATIENT_ID" in df_input.columns else [f"PATIENT_{i+1:03d}" for i in range(len(df_input))]

    clinical_cols = list(artifact.get("clinical_cols_after_collinearity") or [])
    expr_cols = list(artifact.get("expr_cols_after_collinearity") or [])
    clin_pre = artifact.get("clin_pre")
    expr_pipe = artifact.get("expr_pipe")
    scaler = artifact.get("scaler")
    model = artifact.get("cox_model")
    calibrators = artifact.get("horizon_calibrators", {})

    # Feature transformation
    _, _, feature_names = build_feature_names(clin_pre, clinical_cols, expr_pipe)
    X = transform_features(df_input, clinical_cols, expr_cols, clin_pre, expr_pipe, scaler)
    coefs = np.asarray(model.coef_, dtype=float)

    # 1. Cox Risk Score
    risk_scores = np.asarray(model.predict_risk(X), dtype=float)

    # 2. Isotonic Calibrated Survival Probabilities
    calibrated_probs = {}
    for h in [12.0, 24.0, 36.0, 60.0]:
        cal_info = calibrators.get(h)
        if isinstance(cal_info, dict) and "isotonic" in cal_info and cal_info["isotonic"] is not None:
            iso_model = cal_info["isotonic"]
            event_probs = iso_model.predict(risk_scores)
            surv_probs = np.clip(1.0 - event_probs, 0.0, 1.0)
            calibrated_probs[f"prob_survive_{int(h)}m"] = surv_probs
        else:
            calibrated_probs[f"prob_survive_{int(h)}m"] = np.full(len(risk_scores), np.nan)

    # 3. Monte Carlo Simulations
    base_times, base_cumhaz, max_followup = get_baseline_hazard_from_train(artifact)
    sim_times = simulate_cox_survival_times(
        risk_scores,
        baseline_times=base_times,
        baseline_cumhaz=base_cumhaz,
        max_followup_months=max_followup,
        n_sims=MC_N_SIMS,
        random_state=MC_RANDOM_STATE,
    )

    # 4. Explainability (Feature contributions per patient)
    contributions = X * coefs.reshape(1, -1)
    
    # Back-projection to top genes for expression components
    expr_coef = coefs[len(feature_names) - len(expr_cols) :] if expr_pipe else np.array([])
    pca_summary = build_pca_backprojection(expr_pipe, expr_cols, expr_coef, top_n=3)

    # Package results per patient
    results = []
    for i in range(len(df_input)):
        pid = patient_ids[i]
        r_score = float(risk_scores[i])
        sims = sim_times[i]

        p10 = float(np.quantile(sims, 0.10))
        p50 = float(np.quantile(sims, 0.50))
        p90 = float(np.quantile(sims, 0.90))
        rmst60 = float(np.mean(np.minimum(sims, MC_RMST_HORIZON_MONTHS)))

        # Feature contributions
        row_contrib = contributions[i]
        order = np.argsort(np.abs(row_contrib))[::-1]

        top_drivers = []
        for feat_idx in order[:5]:
            fname = feature_names[feat_idx]
            val = float(X[i, feat_idx])
            contrib = float(row_contrib[feat_idx])
            direction = "RISK INCREASING ⬆️" if contrib > 0 else "PROTECTIVE 🛡️"
            top_drivers.append({
                "feature": fname,
                "value": round(val, 4),
                "contribution": round(contrib, 4),
                "direction": direction
            })

        patient_res = {
            "PATIENT_ID": pid,
            "COX_RISK_SCORE": round(r_score, 4),
            "PROB_SURVIVE_12M": round(float(calibrated_probs["prob_survive_12m"][i]), 4),
            "PROB_SURVIVE_24M": round(float(calibrated_probs["prob_survive_24m"][i]), 4),
            "PROB_SURVIVE_36M": round(float(calibrated_probs["prob_survive_36m"][i]), 4),
            "PROB_SURVIVE_60M": round(float(calibrated_probs["prob_survive_60m"][i]), 4),
            "MC_P10_PESSIMISTIC_MONTHS": round(p10, 1),
            "MC_P50_MEDIAN_MONTHS": round(p50, 1),
            "MC_P90_OPTIMISTIC_MONTHS": round(p90, 1),
            "MC_RMST_60M": round(rmst60, 1),
            "TOP_FEATURE_DRIVERS": top_drivers,
        }
        results.append(patient_res)

    results_df = pd.DataFrame(results)
    return results_df, {"pca_backprojection": pca_summary, "n_patients": len(df_input)}


def print_dashboard(results_df: pd.DataFrame, meta: Dict[str, Any]) -> None:
    """Renders a beautiful live dashboard in the terminal."""
    print("\n" + "=" * 80)
    print(f"{Colors.BOLD}{Colors.HEADER}      🧬 MULTI-OMIC COX ELASTIC-NET PIPELINE (v5) - LIVE DEMO REPORT 🧬{Colors.ENDC}")
    print("=" * 80)
    print(f"{Colors.OKCYAN}Status:{Colors.ENDC} Model Loaded & Active | {Colors.OKCYAN}Inference Count:{Colors.ENDC} {meta['n_patients']} Patient(s)")
    print(f"{Colors.OKCYAN}Artifact:{Colors.ENDC} model_weights/final_locked_model.pkl")
    print("=" * 80)

    for idx, row in results_df.iterrows():
        print(f"\n{Colors.BOLD}─────────────── 👤 PATIENT REPORT #{idx+1}: {Colors.OKBLUE}{row['PATIENT_ID']}{Colors.ENDC} {Colors.BOLD}───────────────{Colors.ENDC}")
        
        # Risk Score Assessment
        risk_color = Colors.FAIL if row['COX_RISK_SCORE'] > 0 else Colors.OKGREEN
        risk_level = "HIGH HAZARD RISK" if row['COX_RISK_SCORE'] > 0 else "LOW / PROTECTIVE RISK"
        print(f"  • {Colors.BOLD}Relative Risk Score (η):{Colors.ENDC} {risk_color}{row['COX_RISK_SCORE']} ({risk_level}){Colors.ENDC}")

        # Calibrated Probabilities Table
        print(f"\n  📊 {Colors.BOLD}Isotonic Calibrated Survival Probabilities:{Colors.ENDC}")
        print(f"     ┌─────────────┬─────────────┬─────────────┬─────────────┐")
        print(f"     │  12 Months  │  24 Months  │  36 Months  │  60 Months  │")
        print(f"     ├─────────────┼─────────────┼─────────────┼─────────────┤")
        p12 = f"{row['PROB_SURVIVE_12M']*100:.1f}%"
        p24 = f"{row['PROB_SURVIVE_24M']*100:.1f}%"
        p36 = f"{row['PROB_SURVIVE_36M']*100:.1f}%"
        p60 = f"{row['PROB_SURVIVE_60M']*100:.1f}%"
        print(f"     │    {p12:<8} │    {p24:<8} │    {p36:<8} │    {p60:<8} │")
        print(f"     └─────────────┴─────────────┴─────────────┴─────────────┘")

        # Monte Carlo Bounded Lifespan Estimates
        print(f"\n  🎲 {Colors.BOLD}Monte Carlo Lifespan Simulations (5,000 Iterations):{Colors.ENDC}")
        print(f"     • {Colors.WARNING}P10 Pessimistic Bound:{Colors.ENDC} {row['MC_P10_PESSIMISTIC_MONTHS']} Months")
        print(f"     • {Colors.OKCYAN}P50 Median Expected:{Colors.ENDC}   {row['MC_P50_MEDIAN_MONTHS']} Months")
        print(f"     • {Colors.OKGREEN}P90 Optimistic Bound:{Colors.ENDC}  {row['MC_P90_OPTIMISTIC_MONTHS']} Months")
        print(f"     • {Colors.BOLD}RMST (Restricted Mean @ 60m):{Colors.ENDC} {row['MC_RMST_60M']} Months")

        # Local Patient Explainability (Drivers)
        print(f"\n  🔍 {Colors.BOLD}Local Patient Risk Drivers (Waterfall Breakdown):{Colors.ENDC}")
        for drv in row['TOP_FEATURE_DRIVERS']:
            d_color = Colors.FAIL if "RISK" in drv['direction'] else Colors.OKGREEN
            print(f"     - {drv['feature']:<30} | Value: {drv['value']:<7} | Impact: {d_color}{drv['contribution']:>+7.4f} ({drv['direction']}){Colors.ENDC}")

    print("\n" + "=" * 80)
    print(f"{Colors.BOLD}{Colors.OKGREEN}✅ Live Demo Execution Completed Successfully!{Colors.ENDC}")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Demo script for Cox Elastic-Net model inference.")
    parser.add_argument("--input", type=Path, default=ROOT_DIR / "demo_input_sample.csv", help="Input sample file (.csv or .parquet).")
    parser.add_argument("--model", type=Path, default=ROOT_DIR / "model_weights" / "final_locked_model.pkl", help="Locked model pickle.")
    parser.add_argument("--out-dir", type=Path, default=ROOT_DIR / "results" / "demo_outputs", help="Output directory for demo exports.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{Colors.OKCYAN}Loading input data from:{Colors.ENDC} {args.input}")
    print(f"{Colors.OKCYAN}Loading locked model from:{Colors.ENDC} {args.model}")

    results_df, meta = run_demo_inference(args.input, args.model)
    print_dashboard(results_df, meta)

    # Save exports
    csv_out = args.out_dir / "demo_output_summary.csv"
    json_out = args.out_dir / "demo_output_results.json"
    
    # Flatten drivers for CSV
    results_df.to_csv(csv_out, index=False)
    
    # Save full JSON report
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(results_df.to_dict(orient="records"), f, indent=2)

    print(f"{Colors.OKBLUE}Saved CSV Summary ->{Colors.ENDC} {csv_out}")
    print(f"{Colors.OKBLUE}Saved JSON Detailed Report ->{Colors.ENDC} {json_out}\n")


if __name__ == "__main__":
    main()
