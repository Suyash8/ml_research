#!/usr/bin/env python3
"""
===============================================================================
STEP 06: PCA BACK-PROJECTION & LOCAL XAI WATERFALL DECOMPOSITION
===============================================================================
Input:  results/step_by_step_run/step05_monte_carlo_bounds.pkl
        model_weights/final_locked_model.pkl
Output: results/step_by_step_run/step06_global_gene_weights.csv
        results/step_by_step_run/step06_patient_waterfall_drivers.csv
        results/step_by_step_run/step06_patient_explainability.json
        results/step_by_step_run/step06_patient_explainability.pkl
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

from src.inference.explainability import ExplainabilityModule

def main():
    print("\n" + "="*80)
    print("🔍 STEP 06: PCA BACK-PROJECTION & LOCAL XAI WATERFALL DECOMPOSITION")
    print("="*80)

    input_pkl = ROOT_DIR / "results" / "step_by_step_run" / "step05_monte_carlo_bounds.pkl"
    model_pkl = ROOT_DIR / "model_weights" / "final_locked_model.pkl"
    output_dir = ROOT_DIR / "results" / "step_by_step_run"

    print(f"📥 Loading Step 05 Data: {input_pkl}")
    with open(input_pkl, "rb") as f:
        data_step05 = pickle.load(f)

    patient_ids = data_step05["patient_ids"]
    X_transformed = data_step05["X_transformed"]
    feature_names = data_step05["feature_names"]
    clinical_names = data_step05["clinical_names"]
    expr_cols = data_step05["expr_cols"]
    df_input = data_step05["df_input"]

    print(f"🔒 Unrolling PCA Loadings & Elastic-Net Weights: {model_pkl}")
    
    from src.models.cox_enet import CoxElasticNet
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "CoxElasticNet"):
        setattr(main_module, "CoxElasticNet", CoxElasticNet)

    with open(model_pkl, "rb") as f:
        artifact = pickle.load(f)

    coefs = np.asarray(artifact["cox_model"].coef_, dtype=float)
    pca = artifact["expr_pipe"].named_steps.get("pca") if hasattr(artifact["expr_pipe"], "named_steps") else None
    pca_loadings = getattr(pca, "components_", None) if pca is not None else None

    explain_mod = ExplainabilityModule(
        clinical_feature_names=clinical_names,
        expr_cols=expr_cols,
        coefs=coefs,
        pca_loadings=pca_loadings
    )

    print("⚙️ Computing Global Gene Weights Vector: W_gene = V @ beta_pca (300 genes)...")
    df_gene_weights = explain_mod.compute_gene_risk_weights()
    top_global_risk = df_gene_weights[df_gene_weights["effect_type"] == "risk_increasing"].head(3)["gene_name"].tolist()
    top_global_prot = df_gene_weights[df_gene_weights["effect_type"] == "protective"].head(3)["gene_name"].tolist()
    print(f"   • Top Global Risk Genes:   {', '.join(top_global_risk)}")
    print(f"   • Top Global Protective: {', '.join(top_global_prot)}")

    print("\n⚙️ Computing Local Additive Risk Waterfalls for each patient...")

    n_clin = len(clinical_names)
    patient_explanations = []
    waterfall_rows = []

    for i, pid in enumerate(patient_ids):
        X_clin_patient = X_transformed[i, :n_clin]
        raw_expr = np.zeros(len(expr_cols))
        for g_idx, g_col in enumerate(expr_cols):
            if g_col in df_input.columns:
                raw_expr[g_idx] = float(df_input[g_col].iloc[i])

        exp = explain_mod.explain_patient(pid, X_clin_patient, raw_expr, top_n=5)
        patient_explanations.append(exp)

        top_r = ", ".join(d["feature"] for d in exp["top_risk_drivers"][:2])
        top_p = ", ".join(d["feature"] for d in exp["top_protective_drivers"][:2])
        print(f"   • {pid} -> Top Risk: [{top_r}] | Top Protective: [{top_p}]")

        for rank, driver in enumerate(exp["all_drivers_ranked"], start=1):
            waterfall_rows.append({
                "PATIENT_ID": pid,
                "rank": rank,
                "feature": driver["feature"],
                "type": driver["type"],
                "value": driver["value"],
                "coefficient": driver["coefficient"],
                "contribution": driver["contribution"],
                "direction": driver["direction"]
            })

    # Save PKL
    pkl_out = output_dir / "step06_patient_explainability.pkl"
    out_data = {
        **data_step05,
        "patient_explanations": patient_explanations,
        "global_gene_weights": df_gene_weights.to_dict(orient="records")
    }
    with open(pkl_out, "wb") as f:
        pickle.dump(out_data, f)

    # Save Human-Readable CSVs
    csv_global = output_dir / "step06_global_gene_weights.csv"
    df_gene_weights.to_csv(csv_global, index=False)

    df_waterfall = pd.DataFrame(waterfall_rows)
    csv_waterfall = output_dir / "step06_patient_waterfall_drivers.csv"
    df_waterfall.to_csv(csv_waterfall, index=False)

    # Save Human-Readable JSON
    json_out = output_dir / "step06_patient_explainability.json"
    json_summary = {
        "step": "06_xai_explainability",
        "description": "PCA Back-Projection to Raw Genes & Local Patient Waterfall Risk Drivers",
        "formula": "W_gene = V @ beta_pca ; Delta_eta_g_i = Z_g_i * W_gene_g",
        "n_patients": len(patient_ids),
        "global_top_risk_genes": top_global_risk,
        "global_top_protective_genes": top_global_prot,
        "patient_waterfalls": patient_explanations,
        "readable_global_weights_csv": str(csv_global),
        "readable_waterfall_drivers_csv": str(csv_waterfall)
    }
    with open(json_out, "w") as f:
        json.dump(json_summary, f, indent=2)

    print(f"✅ STEP 06 COMPLETE!")
    print(f"📄 Human-Readable Global Weights CSV: {csv_global}")
    print(f"📄 Human-Readable Patient Drivers CSV: {csv_waterfall}")
    print(f"📄 Human-Readable JSON:                {json_out}\n")

if __name__ == "__main__":
    main()
