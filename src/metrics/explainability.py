from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
from ml_research.src.utils.io import safe_float

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
