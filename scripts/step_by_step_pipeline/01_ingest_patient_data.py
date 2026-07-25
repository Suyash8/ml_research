#!/usr/bin/env python3
"""
===============================================================================
STEP 01: DATA INGESTION & SCHEMATIC VALIDATION
===============================================================================
Input:  scripts/step_by_step_pipeline/00_sample_patient_data.csv
Output: results/step_by_step_run/step01_ingested_data.pkl
        results/step_by_step_run/step01_ingested_data.json
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
    print("📥 STEP 01: DATA INGESTION & SCHEMA VALIDATION")
    print("="*80)

    input_csv = ROOT_DIR / "scripts" / "step_by_step_pipeline" / "00_sample_patient_data.csv"
    output_dir = ROOT_DIR / "results" / "step_by_step_run"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📄 Reading raw input patient data: {input_csv}")
    df = pd.read_csv(input_csv)

    print(f"📊 Input Matrix Dimensions: {df.shape[0]} Rows (Patients) x {df.shape[1]} Columns (Features)")

    patient_ids = df["PATIENT_ID"].astype(str).tolist() if "PATIENT_ID" in df.columns else [f"PATIENT_{i+1:03d}" for i in range(len(df))]
    
    # Identify clinical and gene columns
    gene_cols = [c for c in df.columns if c.startswith("EXPR_")]
    clinical_cols = [c for c in df.columns if c not in gene_cols and c not in ["PATIENT_ID", "OS_MONTHS", "OS_EVENT"]]

    print(f"ℹ️ Extracted {len(clinical_cols)} Clinical Columns and {len(gene_cols)} Gene Expression Columns.")

    out_data = {
        "df_input": df,
        "patient_ids": patient_ids,
        "clinical_cols": clinical_cols,
        "gene_cols": gene_cols,
        "n_patients": len(df)
    }

    # Save PKL
    pkl_out = output_dir / "step01_ingested_data.pkl"
    with open(pkl_out, "wb") as f:
        pickle.dump(out_data, f)

    # Save JSON summary
    json_out = output_dir / "step01_ingested_data.json"
    json_summary = {
        "step": "01_ingest_patient_data",
        "n_patients": len(df),
        "patient_ids": patient_ids,
        "n_clinical_features": len(clinical_cols),
        "n_gene_features": len(gene_cols),
        "output_file": str(pkl_out)
    }
    with open(json_out, "w") as f:
        json.dump(json_summary, f, indent=2)

    print(f"✅ STEP 01 COMPLETE!")
    print(f"💾 Saved Intermediate Output: {pkl_out}")
    print(f"💾 Saved JSON Summary:        {json_out}\n")

if __name__ == "__main__":
    main()
