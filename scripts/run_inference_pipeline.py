#!/usr/bin/env python3
"""
===============================================================================
CLI RUNNER: Multi-Omic Cox Elastic-Net Inference Pipeline
===============================================================================
Executes real-time predictions and explainability analysis on input clinical &
genomic data files using the modular `MultiOmicInferencePipeline`.

Usage:
    python scripts/run_inference_pipeline.py --input demo_input_sample.csv
    python scripts/run_inference_pipeline.py --input demo_input_sample.csv --output results/demo_predictions.json
===============================================================================
"""

import argparse
import json
import sys
from pathlib import Path

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from src.inference.pipeline import MultiOmicInferencePipeline


def main():
    parser = argparse.ArgumentParser(description="Run Multi-Omic Cox Elastic-Net Inference Pipeline")
    parser.add_argument(
        "--input",
        type=str,
        default="demo_input_sample.csv",
        help="Path to input patient CSV or Parquet file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model_weights/final_locked_model.pkl",
        help="Path to locked model artifact pickle file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/demo_outputs/pipeline_inference_results.json",
        help="Path to save output prediction JSON file"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT_DIR / input_path

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT_DIR / model_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT_DIR / output_path

    print("===============================================================================")
    print("🚀 INITIATING MULTI-OMIC INFERENCE & EXPLAINABILITY PIPELINE")
    print("===============================================================================")
    print(f"📥 Input File: {input_path}")
    print(f"🔒 Locked Model: {model_path}")

    # Load input data
    if input_path.suffix.lower() == ".parquet":
        df_input = pd.read_parquet(input_path)
    else:
        df_input = pd.read_csv(input_path)

    print(f"📊 Loaded {len(df_input)} patient record(s).")

    # Initialize Inference Pipeline
    pipeline = MultiOmicInferencePipeline(model_path)

    # 1. Run Unified Predictions
    print("\n⚙️ Running Risk Scoring, Isotonic Calibration & Monte Carlo Simulations...")
    df_results = pipeline.predict(df_input)

    # 2. Run Detailed Patient Explainability
    print("🔍 Computing PCA Back-Projection Gene Drivers & Patient Waterfalls...")
    explanations = pipeline.explain(df_input, top_n_drivers=5)

    print("\n===============================================================================")
    print("📋 SUMMARY PREDICTION RESULTS")
    print("===============================================================================")
    print(df_results.to_string(index=False))

    # Save outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_output_path = output_path.with_suffix(".csv")
    df_results.to_csv(csv_output_path, index=False)
    
    # Save JSON with detailed explanations
    full_output = {
        "predictions_summary": df_results.to_dict(orient="records"),
        "detailed_explanations": explanations
    }
    with open(output_path, "w") as f:
        json.dump(full_output, f, indent=2)

    print("\n===============================================================================")
    print(f"✅ INFERENCE COMPLETE!")
    print(f"📄 CSV Output Saved:  {csv_output_path}")
    print(f"📄 JSON Output Saved: {output_path}")
    print("===============================================================================")


if __name__ == "__main__":
    main()
