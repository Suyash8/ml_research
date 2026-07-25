#!/usr/bin/env python3
"""
===============================================================================
STEP 02: FEATURE TRANSFORMATIONS & PCA DIMENSIONALITY REDUCTION
===============================================================================
Input:  results/step_by_step_run/step01_ingested_data.pkl
        model_weights/final_locked_model.pkl
Output: results/step_by_step_run/step02_transformed_clinical.csv
        results/step_by_step_run/step02_transformed_pca.csv
        results/step_by_step_run/step02_transformed_combined_features.csv
        results/step_by_step_run/step02_transformed_features.json
        results/step_by_step_run/step02_transformed_features.pkl
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

from src.features.preprocessing import transform_features
from src.metrics.explainability import build_feature_names

def main():
    print("\n" + "="*80)
    print("🛠️ STEP 02: FEATURE TRANSFORMATIONS & PCA DIMENSIONALITY REDUCTION")
    print("="*80)

    input_pkl = ROOT_DIR / "results" / "step_by_step_run" / "step01_ingested_data.pkl"
    model_pkl = ROOT_DIR / "model_weights" / "final_locked_model.pkl"
    output_dir = ROOT_DIR / "results" / "step_by_step_run"

    print(f"📥 Loading Step 01 Data:  {input_pkl}")
    with open(input_pkl, "rb") as f:
        data_step01 = pickle.load(f)

    df_input = data_step01["df_input"]
    patient_ids = data_step01["patient_ids"]

    # Register CoxElasticNet for unpickling
    from src.models.cox_enet import CoxElasticNet
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "CoxElasticNet"):
        setattr(main_module, "CoxElasticNet", CoxElasticNet)

    with open(model_pkl, "rb") as f:
        artifact = pickle.load(f)

    clinical_cols = list(artifact["clinical_cols_after_collinearity"])
    expr_cols = list(artifact["expr_cols_after_collinearity"])
    clin_pre = artifact["clin_pre"]
    expr_pipe = artifact["expr_pipe"]
    scaler = artifact["scaler"]

    print("⚙️ Applying Clinical Preprocessor (Median Impute + One-Hot Encoding + StandardScaler)...")
    print("⚙️ Applying Genomic Pipeline (log2(x+1) + Gram-Schmidt + PCA 50 components)...")

    clin_names, expr_names, feature_names = build_feature_names(clin_pre, clinical_cols, expr_pipe)
    X_transformed = transform_features(df_input, clinical_cols, expr_cols, clin_pre, expr_pipe, scaler)

    n_clin = len(clin_names)
    X_clin = X_transformed[:, :n_clin]
    X_pca = X_transformed[:, n_clin:]

    print(f"📊 Transformed Clinical Matrix (X_clin): [N={len(patient_ids)} x P_clin={X_clin.shape[1]}]")
    print(f"📊 Transformed Latent PCA Matrix (X_pca): [N={len(patient_ids)} x P_pca={X_pca.shape[1]}]")
    print(f"📊 Final Concatenated Model Matrix (X):   [N={len(patient_ids)} x Total={X_transformed.shape[1]}]")

    # Save PKL
    out_data = {
        "df_input": df_input,
        "patient_ids": patient_ids,
        "X_transformed": X_transformed,
        "X_clin": X_clin,
        "X_pca": X_pca,
        "feature_names": feature_names,
        "clinical_names": clin_names,
        "expr_names": expr_names,
        "expr_cols": expr_cols
    }
    pkl_out = output_dir / "step02_transformed_features.pkl"
    with open(pkl_out, "wb") as f:
        pickle.dump(out_data, f)

    # Save Human-Readable CSVs
    df_clin = pd.DataFrame(X_clin, columns=clin_names)
    df_clin.insert(0, "PATIENT_ID", patient_ids)
    csv_clin = output_dir / "step02_transformed_clinical.csv"
    df_clin.to_csv(csv_clin, index=False)

    df_pca = pd.DataFrame(X_pca, columns=expr_names)
    df_pca.insert(0, "PATIENT_ID", patient_ids)
    csv_pca = output_dir / "step02_transformed_pca.csv"
    df_pca.to_csv(csv_pca, index=False)

    df_combined = pd.DataFrame(X_transformed, columns=feature_names)
    df_combined.insert(0, "PATIENT_ID", patient_ids)
    csv_combined = output_dir / "step02_transformed_combined_features.csv"
    df_combined.to_csv(csv_combined, index=False)

    # Save Human-Readable JSON
    json_out = output_dir / "step02_transformed_features.json"
    json_summary = {
        "step": "02_feature_transformation",
        "description": "Feature Preprocessing and PCA Latent Projection",
        "n_patients": len(patient_ids),
        "n_clinical_transformed_features": X_clin.shape[1],
        "n_pca_components": X_pca.shape[1],
        "total_model_features": X_transformed.shape[1],
        "transformed_clinical_columns": clin_names,
        "pca_component_columns": expr_names,
        "readable_clinical_csv": str(csv_clin),
        "readable_pca_csv": str(csv_pca),
        "readable_combined_csv": str(csv_combined)
    }
    with open(json_out, "w") as f:
        json.dump(json_summary, f, indent=2)

    print(f"✅ STEP 02 COMPLETE!")
    print(f"📄 Human-Readable Clinical CSV: {csv_clin}")
    print(f"📄 Human-Readable PCA CSV:      {csv_pca}")
    print(f"📄 Human-Readable Combined CSV: {csv_combined}")
    print(f"📄 Human-Readable JSON:         {json_out}\n")

if __name__ == "__main__":
    main()
