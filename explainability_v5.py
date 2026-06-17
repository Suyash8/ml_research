"""Explainability companion for the final v5 Cox Elastic-Net pipeline.

This script does not retrain the model or modify the original v5 code.
It loads the locked v5 artifact, reconstructs the transformed feature space,
and exports global, group-level, PCA back-projection, and patient-level
explanations that align with the final v5 outputs.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import cox_enet_calibrated_mc_pipeline_v5 as v5


BASE = Path("/home/illionar/Projects/ml_research")
INPUT_PATH = BASE / "data" / "preprocessed_cleaned" / "patient_multiomic_cleaned.parquet"
MODEL_PATH = BASE / "data" / "model_outputs" / "cox_enet_calibrated_mc_outputs_v5" / "final_locked_model.pkl"
PREDICTIONS_PATH = BASE / "data" / "model_outputs" / "cox_enet_calibrated_mc_outputs_v5" / "main_predictions.csv"
SOURCE_METRICS_PATH = BASE / "data" / "model_outputs" / "cox_enet_calibrated_mc_outputs_v5" / "metrics.json"
DEFAULT_OUT_DIR = BASE / "data" / "model_outputs" / "cox_enet_calibrated_mc_outputs_v5_explainability"


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def safe_float(x: Any) -> float:
    try:
        value = float(x)
    except Exception:
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def load_artifact(model_path: Path) -> Dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Locked model artifact not found: {model_path}")
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "CoxElasticNet"):
        setattr(main_module, "CoxElasticNet", v5.CoxElasticNet)
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


def build_feature_names(
    clin_pre,
    clinical_cols: Sequence[str],
    expr_pipe,
) -> Tuple[List[str], List[str], List[str]]:
    clinical_names: List[str] = []
    expression_names: List[str] = []

    if clin_pre is not None and clinical_cols:
        try:
            clinical_names = [str(name) for name in clin_pre.get_feature_names_out(list(clinical_cols))]
        except Exception:
            clinical_names = [str(col) for col in clinical_cols]

    n_expr_components = 0
    if expr_pipe is not None and hasattr(expr_pipe, "named_steps"):
        pca = expr_pipe.named_steps.get("pca")
        if pca is not None:
            n_expr_components = int(getattr(pca, "n_components_", getattr(pca, "n_components", 0)) or 0)
    expression_names = [f"EXPR_PC{i + 1:02d}" for i in range(n_expr_components)]

    return clinical_names, expression_names, clinical_names + expression_names


def make_global_importance(feature_names: Sequence[str], coef: np.ndarray) -> pd.DataFrame:
    rows = []
    for feature_name, value in zip(feature_names, coef):
        value = safe_float(value)
        rows.append(
            {
                "feature_name": str(feature_name),
                "group": "expression" if str(feature_name).startswith("EXPR_PC") else "clinical",
                "coefficient": value,
                "abs_coefficient": safe_float(abs(value)),
                "direction": "risk_increasing" if value > 0 else "risk_decreasing" if value < 0 else "neutral",
            }
        )
    return pd.DataFrame(rows).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def make_group_summary(feature_table: pd.DataFrame) -> pd.DataFrame:
    if feature_table.empty:
        return pd.DataFrame()

    rows = []
    for group_name, group_df in feature_table.groupby("group", dropna=False):
        coef = group_df["coefficient"].to_numpy(dtype=float)
        rows.append(
            {
                "group": str(group_name),
                "n_features": int(len(group_df)),
                "sum_coefficient": safe_float(np.sum(coef)),
                "sum_abs_coefficient": safe_float(np.sum(np.abs(coef))),
                "mean_abs_coefficient": safe_float(np.mean(np.abs(coef))),
                "max_abs_coefficient": safe_float(np.max(np.abs(coef))),
            }
        )
    return pd.DataFrame(rows).sort_values("sum_abs_coefficient", ascending=False).reset_index(drop=True)


def build_pca_backprojection(
    expr_pipe,
    expr_cols: Sequence[str],
    expr_coef: np.ndarray,
    top_n: int,
) -> pd.DataFrame:
    if expr_pipe is None or not expr_cols or expr_coef.size == 0:
        return pd.DataFrame()

    pca = expr_pipe.named_steps.get("pca") if hasattr(expr_pipe, "named_steps") else None
    if pca is None or not hasattr(pca, "components_"):
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    components = np.asarray(pca.components_, dtype=float)
    for pc_idx, pc_loading in enumerate(components):
        if pc_idx >= len(expr_coef):
            break
        pc_weight = safe_float(expr_coef[pc_idx])
        weighted = pc_loading * pc_weight
        top_indices = np.argsort(np.abs(weighted))[::-1][:top_n]
        for rank, gene_idx in enumerate(top_indices, start=1):
            rows.append(
                {
                    "pc_index": int(pc_idx + 1),
                    "pc_name": f"EXPR_PC{pc_idx + 1:02d}",
                    "pc_coefficient": pc_weight,
                    "gene_name": str(expr_cols[gene_idx]),
                    "gene_loading": safe_float(pc_loading[gene_idx]),
                    "risk_weighted_loading": safe_float(weighted[gene_idx]),
                    "rank_within_pc": int(rank),
                }
            )
    return pd.DataFrame(rows)


def make_patient_explanations(
    patient_df: pd.DataFrame,
    X: np.ndarray,
    feature_names: Sequence[str],
    coef: np.ndarray,
    top_features: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if patient_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    contributions = X * coef.reshape(1, -1)
    summary_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []

    clinical_idx = [i for i, name in enumerate(feature_names) if not str(name).startswith("EXPR_PC")]
    expr_idx = [i for i, name in enumerate(feature_names) if str(name).startswith("EXPR_PC")]

    for row_idx, (_, row) in enumerate(patient_df.iterrows()):
        row_contrib = contributions[row_idx]
        order = np.argsort(np.abs(row_contrib))[::-1]
        top_idx = order[:top_features]

        pos_idx = np.where(row_contrib > 0)[0]
        neg_idx = np.where(row_contrib < 0)[0]
        top_positive_idx = int(pos_idx[np.argmax(row_contrib[pos_idx])]) if len(pos_idx) else -1
        top_negative_idx = int(neg_idx[np.argmin(row_contrib[neg_idx])]) if len(neg_idx) else -1

        summary_rows.append(
            {
                "PATIENT_ID": str(row["PATIENT_ID"]),
                "risk_score": safe_float(row.get("risk_score", np.nan)),
                "OS_MONTHS": safe_float(row.get("OS_MONTHS", np.nan)),
                "OS_EVENT": int(row.get("OS_EVENT", 0)) if pd.notna(row.get("OS_EVENT", np.nan)) else None,
                "recomputed_log_risk": safe_float(np.sum(row_contrib)),
                "top_positive_feature": str(feature_names[top_positive_idx]) if top_positive_idx >= 0 else "",
                "top_positive_contribution": safe_float(row_contrib[top_positive_idx]) if top_positive_idx >= 0 else float("nan"),
                "top_negative_feature": str(feature_names[top_negative_idx]) if top_negative_idx >= 0 else "",
                "top_negative_contribution": safe_float(row_contrib[top_negative_idx]) if top_negative_idx >= 0 else float("nan"),
                "clinical_contribution_sum": safe_float(np.sum(row_contrib[clinical_idx])) if clinical_idx else float("nan"),
                "expression_contribution_sum": safe_float(np.sum(row_contrib[expr_idx])) if expr_idx else float("nan"),
            }
        )

        for rank, feature_idx in enumerate(top_idx, start=1):
            detail_rows.append(
                {
                    "PATIENT_ID": str(row["PATIENT_ID"]),
                    "rank": int(rank),
                    "feature_name": str(feature_names[feature_idx]),
                    "feature_value": safe_float(X[row_idx, feature_idx]),
                    "coefficient": safe_float(coef[feature_idx]),
                    "contribution": safe_float(row_contrib[feature_idx]),
                    "direction": "risk_increasing" if row_contrib[feature_idx] > 0 else "risk_decreasing" if row_contrib[feature_idx] < 0 else "neutral",
                    "is_expression_pc": bool(str(feature_names[feature_idx]).startswith("EXPR_PC")),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def write_markdown_report(
    out_dir: Path,
    metrics: Dict[str, Any],
    feature_table: pd.DataFrame,
    group_summary: pd.DataFrame,
    pca_summary: pd.DataFrame,
) -> None:
    lines: List[str] = [
        "# v5 Explainability Report",
        "",
        "This report is generated from the locked v5 Cox Elastic-Net artifact.",
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
    parser = argparse.ArgumentParser(description="Generate explainability artifacts for the v5 Cox ENet model.")
    parser.add_argument("--input", type=Path, default=INPUT_PATH, help="Cleaned input parquet used by v5.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Locked v5 model artifact.")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH, help="v5 predictions CSV.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for XAI outputs.")
    parser.add_argument("--top-features", type=int, default=12, help="Top features to keep per patient.")
    parser.add_argument("--top-genes-per-pc", type=int, default=10, help="Top genes to keep per PCA component.")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact = load_artifact(args.model)
    df = v5.prepare_dataframe(args.input)
    pred_df = load_predictions(args.predictions)
    source_metrics = load_source_metrics(SOURCE_METRICS_PATH)

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

    X = v5.transform_features(df, clinical_cols, expr_cols, clin_pre, expr_pipe, scaler)
    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature-name mismatch: transformed matrix has {X.shape[1]} columns, but {len(feature_names)} names were generated."
        )

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
        "source_metrics": str(SOURCE_METRICS_PATH),
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