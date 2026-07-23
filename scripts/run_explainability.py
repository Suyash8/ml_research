import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ml_research.src.data.loader import prepare_dataframe
from ml_research.src.features.preprocessing import transform_features
from ml_research.src.metrics.explainability import (
    build_feature_names,
    make_global_importance,
    make_group_summary,
    build_pca_backprojection,
    make_patient_explanations,
)
from ml_research.src.utils.io import save_json, safe_float

def load_artifact(model_path: Path) -> Dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Locked model artifact not found: {model_path}")
    
    # Ensure CoxElasticNet is in sys.modules or __main__ since pickle might look for it there
    from ml_research.src.models.cox_enet import CoxElasticNet
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "CoxElasticNet"):
        setattr(main_module, "CoxElasticNet", CoxElasticNet)
        
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    if not isinstance(artifact, dict):
        raise TypeError("Unexpected artifact format: expected a dict")
    return artifact

def load_predictions(predictions_path: Path) -> pd.DataFrame:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Prediction export not found: {predictions_path}")
    return pd.read_csv(predictions_path)

def load_source_metrics(metrics_path: Path) -> Dict[str, Any]:
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))

def write_markdown_report(
    out_dir: Path,
    metrics: Dict[str, Any],
    feature_table: pd.DataFrame,
    group_summary: pd.DataFrame,
    pca_summary: pd.DataFrame,
) -> None:
    lines: List[str] = [
        "# Explainability Report",
        "",
        "This report is generated from the locked Cox Elastic-Net artifact.",
        "It explains the transformed feature space, not raw gene effects directly.",
        "",
        "## Model Context",
        f"- train C-index: {metrics.get('c_index_train', float('nan')):.4f}",
        f"- calibration C-index: {metrics.get('c_index_calibration', float('nan')):.4f}",
        f"- test C-index: {metrics.get('c_index_test', float('nan')):.4f}",
        "",
        "## Global Drivers",
    ]

    for _, row in feature_table.head(12).iterrows():
        lines.append(f"- {row['feature_name']}: coef={row['coefficient']:.4f} ({row['direction']})")

    lines += ["", "## Group Summary"]
    for _, row in group_summary.iterrows():
        lines.append(f"- {row['group']}: n={int(row['n_features'])}, sum_abs_coef={row['sum_abs_coefficient']:.4f}")

    if not pca_summary.empty:
        lines += ["", "## Expression Back-Projection"]
        for _, row in pca_summary.head(20).iterrows():
            lines.append(
                f"- {row['pc_name']} -> {row['gene_name']}: loading={row['gene_loading']:.4f}, weighted={row['risk_weighted_loading']:.4f}"
            )

    lines += [
        "",
        "## Interpretation Notes",
        "- Clinical features are directly interpretable in transformed form.",
        "- Expression coefficients live on PCA components; gene-level meaning comes from back-projection.",
        "- Patient-level contributions are additive on the Cox log-risk scale.",
        "- The Monte Carlo survival outputs remain uncertainty summaries, not exact lifespan predictions.",
    ]
    (out_dir / "xai_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate explainability artifacts for the Cox ENet model.")
    parser.add_argument("--input", type=Path, default=Path("data/preprocessed_cleaned/patient_multiomic_cleaned.parquet"), help="Cleaned input parquet.")
    parser.add_argument("--model", type=Path, default=Path("model_weights/final_locked_model.pkl"), help="Locked model artifact.")
    parser.add_argument("--predictions", type=Path, default=Path("results/cox_enet_calibrated_mc_outputs_v5/main_predictions.csv"), help="Predictions CSV.")
    parser.add_argument("--source-metrics", type=Path, default=Path("results/cox_enet_calibrated_mc_outputs_v5/metrics.json"), help="Metrics JSON.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/explainability"), help="Directory for XAI outputs.")
    parser.add_argument("--top-features", type=int, default=1000, help="Top features to keep per patient.")
    parser.add_argument("--top-genes-per-pc", type=int, default=10, help="Top genes to keep per PCA component.")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact = load_artifact(args.model)
    df = prepare_dataframe(args.input)
    pred_df = load_predictions(args.predictions)
    source_metrics = load_source_metrics(args.source_metrics)

    clinical_cols = list(
        artifact.get("clinical_cols_after_collinearity")
        or artifact.get("clinical_cols_before_collinearity")
        or []
    )
    expr_cols = list(artifact.get("expr_cols_after_collinearity") or artifact.get("expr_cols_before_collinearity") or [])
    clin_pre = artifact.get("clin_pre")
    expr_pipe = artifact.get("expr_pipe")
    scaler = artifact.get("scaler")
    model = artifact.get("cox_model")

    if scaler is None or model is None:
        raise ValueError("Locked artifact is missing the fitted scaler or Cox model.")

    clinical_names, expression_names, feature_names = build_feature_names(clin_pre, clinical_cols, expr_pipe)
    if not feature_names:
        raise ValueError("Unable to recover transformed feature names from the locked artifact.")

    X = transform_features(df, clinical_cols, expr_cols, clin_pre, expr_pipe, scaler)
    
    risk_scores = np.asarray(model.predict_risk(X), dtype=float) if hasattr(model, "predict_risk") else np.asarray(X @ np.asarray(model.coef_, dtype=float), dtype=float)

    transformed_df = pd.DataFrame(X, columns=feature_names)
    transformed_df.insert(0, "PATIENT_ID", df["PATIENT_ID"].astype(str).to_numpy())
    transformed_df.insert(1, "OS_MONTHS", df["OS_MONTHS"].to_numpy())
    transformed_df.insert(2, "OS_EVENT", df["OS_EVENT"].to_numpy())
    transformed_df["risk_score_recomputed"] = risk_scores

    feature_table = make_global_importance(feature_names, np.asarray(model.coef_, dtype=float))
    group_summary = make_group_summary(feature_table)

    expr_coef = np.asarray(model.coef_, dtype=float)[len(clinical_names) :]
    pca_summary = build_pca_backprojection(expr_pipe, expr_cols, expr_coef, top_n=int(args.top_genes_per_pc))

    test_pred = pred_df[pred_df["split"].astype(str).str.lower() == "test"].copy()
    patient_frame = pd.merge(
        test_pred[["PATIENT_ID", "risk_score", "OS_MONTHS", "OS_EVENT"]],
        transformed_df[["PATIENT_ID"] + feature_names],
        on="PATIENT_ID",
        how="inner",
    )
    if patient_frame.empty:
        raise ValueError("No test rows were matched between predictions and transformed features.")

    xai_feature_matrix = patient_frame[feature_names].to_numpy(dtype=float)
    summary_df, detail_df = make_patient_explanations(
        patient_frame[["PATIENT_ID", "risk_score", "OS_MONTHS", "OS_EVENT"]].copy(),
        xai_feature_matrix,
        feature_names,
        np.asarray(model.coef_, dtype=float),
        top_features=int(args.top_features),
    )

    summary_df = pd.merge(summary_df, test_pred, on="PATIENT_ID", how="left", suffixes=("", "_pred"))

    feature_table.to_csv(out_dir / "xai_global_feature_importance.csv", index=False)
    group_summary.to_csv(out_dir / "xai_group_summary.csv", index=False)
    pca_summary.to_csv(out_dir / "xai_pca_component_loadings.csv", index=False)
    summary_df.to_csv(out_dir / "xai_patient_summary_test.csv", index=False)
    detail_df.to_csv(out_dir / "xai_patient_feature_contributions_test.csv", index=False)

    xai_metrics: Dict[str, Any] = {
        "source_model_artifact": str(args.model),
        "source_predictions": str(args.predictions),
        "source_input": str(args.input),
        "source_metrics": str(args.source_metrics),
        "n_rows_total": int(len(df)),
        "n_rows_test_explained": int(len(summary_df)),
        "n_transformed_features": int(len(feature_names)),
        "n_clinical_transformed_features": int(len(clinical_names)),
        "n_expression_pca_features": int(len(expression_names)),
        "top_features_per_patient": int(args.top_features),
        "top_genes_per_pc": int(args.top_genes_per_pc),
        "risk_score_mean": safe_float(np.mean(risk_scores)),
        "risk_score_median": safe_float(np.median(risk_scores)),
        "risk_score_recomputed_max_abs_diff": safe_float(np.max(np.abs(risk_scores - np.asarray(transformed_df["risk_score_recomputed"], dtype=float)))),
    }
    for key in ["c_index_train", "c_index_calibration", "c_index_test", "cv_best_mean_c_index", "cv_best_std_c_index"]:
        if key in source_metrics:
            xai_metrics[key] = source_metrics[key]
    if "horizon_metrics" in source_metrics:
        xai_metrics["source_horizon_metrics"] = source_metrics["horizon_metrics"]
    save_json(out_dir / "xai_metrics.json", xai_metrics)

    provenance = {
        "model_path": str(args.model),
        "input_path": str(args.input),
        "predictions_path": str(args.predictions),
        "output_dir": str(out_dir),
        "clinical_cols": clinical_cols,
        "expr_cols": expr_cols,
        "feature_names": feature_names,
    }
    save_json(out_dir / "xai_provenance.json", provenance)
    write_markdown_report(out_dir, xai_metrics, feature_table, group_summary, pca_summary)

    print("Done.")
    print(f"Explainability outputs -> {out_dir.resolve()}")

if __name__ == "__main__":
    main()
