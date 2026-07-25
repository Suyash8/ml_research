#!/usr/bin/env bash

set -e

echo "==============================================================================="
echo "🚀 EXECUTING COMPLETE STEP-BY-STEP MODULAR PIPELINE RUNNER"
echo "==============================================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT_DIR"

uv run python "$SCRIPT_DIR/01_ingest_patient_data.py"
uv run python "$SCRIPT_DIR/02_feature_transformation.py"
uv run python "$SCRIPT_DIR/03_cox_risk_scoring.py"
uv run python "$SCRIPT_DIR/04_isotonic_calibration.py"
uv run python "$SCRIPT_DIR/05_monte_carlo_simulation.py"
uv run python "$SCRIPT_DIR/06_xai_explainability.py"
uv run python "$SCRIPT_DIR/07_generate_final_dossier.py"

echo "==============================================================================="
echo "🎉 ALL PIPELINE STEPS EXECUTED AND VERIFIED SUCCESSFULLY!"
echo "📂 Intermediate & Final Step Runs Saved In: $ROOT_DIR/results/step_by_step_run/"
echo "==============================================================================="
