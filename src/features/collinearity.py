from typing import Any, Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from ml_research.src.utils.config import COLLINEARITY_MAX_FULL_CORR_FEATURES, COLLINEARITY_THRESHOLD

def _empty_drop_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["group", "feature_dropped", "anchor_feature", "abs_correlation", "threshold", "method"]
    )

def _drop_by_full_correlation(
    numeric_df: pd.DataFrame, threshold: float, group_name: str
) -> Tuple[Set[str], pd.DataFrame, Dict[str, Any]]:
    abs_corr = numeric_df.corr(method="pearson").abs()
    if abs_corr.empty:
        return set(), _empty_drop_table(), {"method": "full_correlation_matrix", "n_pairs_above_threshold": 0}

    upper_mask = np.triu(np.ones(abs_corr.shape, dtype=bool), k=1)
    upper = abs_corr.where(upper_mask)
    upper_arr = upper.to_numpy(dtype=float)
    ri, ci = np.where(np.isfinite(upper_arr) & (upper_arr > threshold))
    high_pairs = sorted(
        [(str(upper.index[i]), str(upper.columns[j]), float(upper_arr[i, j])) for i, j in zip(ri, ci)],
        key=lambda t: t[2],
        reverse=True,
    )

    drop_rows: List[Dict[str, Any]] = []
    drop_set: Set[str] = set()
    for col in upper.columns:
        high = upper[col][upper[col] > threshold].dropna()
        if high.empty:
            continue
        anchor = str(high.idxmax())
        corr_val = float(high.max())
        drop_set.add(str(col))
        drop_rows.append(
            {
                "group": group_name,
                "feature_dropped": str(col),
                "anchor_feature": anchor,
                "abs_correlation": corr_val,
                "threshold": float(threshold),
                "method": "full_correlation_matrix",
            }
        )

    preview = [
        {"feature_a": a, "feature_b": b, "abs_correlation": c}
        for a, b, c in high_pairs[:20]
    ]
    return drop_set, pd.DataFrame(drop_rows), {
        "method": "full_correlation_matrix",
        "n_pairs_above_threshold": int(len(high_pairs)),
        "top_high_correlation_pairs_preview": preview,
    }

def _drop_by_incremental_correlation(
    numeric_df: pd.DataFrame, threshold: float, group_name: str
) -> Tuple[Set[str], pd.DataFrame, Dict[str, Any]]:
    cols = list(numeric_df.columns)
    if len(cols) < 2:
        return set(), _empty_drop_table(), {"method": "incremental_against_kept", "n_dropped": 0}

    X = np.asarray(numeric_df.values, dtype=float)
    medians = np.nanmedian(X, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    bad = ~np.isfinite(X)
    if np.any(bad):
        r, c = np.where(bad)
        X[r, c] = medians[c]

    X = X - np.mean(X, axis=0, keepdims=True)
    norms = np.linalg.norm(X, axis=0)
    kept_idx: List[int] = []
    drop_rows: List[Dict[str, Any]] = []

    for j, col_name in enumerate(cols):
        if not kept_idx:
            kept_idx.append(j)
            continue
        if norms[j] < 1e-12:
            kept_idx.append(j)
            continue

        k = np.asarray(kept_idx, dtype=int)
        k = k[norms[k] >= 1e-12]
        if len(k) == 0:
            kept_idx.append(j)
            continue

        corr_vec = np.abs((X[:, k].T @ X[:, j]) / (norms[k] * norms[j]))
        best_idx = int(np.argmax(corr_vec))
        best_corr = float(corr_vec[best_idx])

        if best_corr > threshold:
            drop_rows.append(
                {
                    "group": group_name,
                    "feature_dropped": str(col_name),
                    "anchor_feature": str(cols[int(k[best_idx])]),
                    "abs_correlation": best_corr,
                    "threshold": float(threshold),
                    "method": "incremental_against_kept",
                }
            )
        else:
            kept_idx.append(j)

    drop_set = {str(r["feature_dropped"]) for r in drop_rows}
    return drop_set, pd.DataFrame(drop_rows), {
        "method": "incremental_against_kept",
        "n_dropped": int(len(drop_rows)),
    }

def drop_collinear_features(
    df_reference: pd.DataFrame,
    columns: List[str],
    threshold: float,
    group_name: str,
) -> Tuple[List[str], pd.DataFrame, Dict[str, Any]]:
    if not columns:
        return [], _empty_drop_table(), {
            "group": group_name,
            "threshold": float(threshold),
            "input_feature_count": 0,
            "numeric_evaluated_count": 0,
            "dropped_count": 0,
            "kept_count": 0,
            "skipped_non_numeric_or_too_sparse": [],
            "method_details": {"method": "none"},
        }

    numeric_data: Dict[str, pd.Series] = {}
    skipped_sparse: List[str] = []
    skipped_constant: List[str] = []

    for col in columns:
        s = pd.to_numeric(df_reference[col], errors="coerce")
        if int(s.notna().sum()) < 2:
            skipped_sparse.append(col)
            continue
        std = float(s.std(skipna=True))
        if (not np.isfinite(std)) or std < 1e-12:
            skipped_constant.append(col)
            continue
        numeric_data[col] = s.astype(float)

    numeric_df = pd.DataFrame(numeric_data)
    if numeric_df.shape[1] < 2:
        return list(columns), _empty_drop_table(), {
            "group": group_name,
            "threshold": float(threshold),
            "input_feature_count": int(len(columns)),
            "numeric_evaluated_count": int(numeric_df.shape[1]),
            "dropped_count": 0,
            "kept_count": int(len(columns)),
            "skipped_non_numeric_or_too_sparse": skipped_sparse,
            "skipped_constant_or_near_constant": skipped_constant,
            "method_details": {"method": "insufficient_numeric_features"},
        }

    if numeric_df.shape[1] <= COLLINEARITY_MAX_FULL_CORR_FEATURES:
        drop_set, drop_table, method_details = _drop_by_full_correlation(numeric_df, threshold, group_name)
    else:
        drop_set, drop_table, method_details = _drop_by_incremental_correlation(numeric_df, threshold, group_name)

    kept_columns = [c for c in columns if c not in drop_set]
    if drop_table.empty:
        drop_table = _empty_drop_table()

    summary = {
        "group": group_name,
        "threshold": float(threshold),
        "input_feature_count": int(len(columns)),
        "numeric_evaluated_count": int(numeric_df.shape[1]),
        "dropped_count": int(len(drop_set)),
        "kept_count": int(len(kept_columns)),
        "dropped_features": sorted(drop_set),
        "skipped_non_numeric_or_too_sparse": skipped_sparse,
        "skipped_constant_or_near_constant": skipped_constant,
        "method_details": method_details,
    }
    return kept_columns, drop_table, summary

def apply_collinearity_filter(
    df_train: pd.DataFrame,
    clinical_cols: List[str],
    expr_cols: List[str],
    threshold: float = COLLINEARITY_THRESHOLD,
) -> Tuple[List[str], List[str], pd.DataFrame, Dict[str, Any]]:
    expr_kept, expr_drop_table, expr_summary = drop_collinear_features(
        df_reference=df_train, columns=expr_cols, threshold=threshold, group_name="expression"
    )

    clin_numeric: List[str] = []
    clin_non_numeric: List[str] = []
    for col in clinical_cols:
        s = pd.to_numeric(df_train[col], errors="coerce")
        if int(s.notna().sum()) >= 2:
            clin_numeric.append(col)
        else:
            clin_non_numeric.append(col)

    clin_kept_numeric, clin_drop_table, clin_summary = drop_collinear_features(
        df_reference=df_train, columns=clin_numeric, threshold=threshold, group_name="clinical_numeric"
    )
    clin_dropped = set(clin_summary.get("dropped_features", []))
    clinical_final = [c for c in clinical_cols if c not in clin_dropped]

    all_tables = [t for t in (expr_drop_table, clin_drop_table) if not t.empty]
    dropped_df = (
        pd.concat(all_tables, axis=0, ignore_index=True)
        .sort_values(["group", "abs_correlation", "feature_dropped"], ascending=[True, False, True])
        if all_tables
        else _empty_drop_table()
    )

    summary = {
        "threshold_abs_pearson": float(threshold),
        "fit_reference": "train_split_only",
        "expression": expr_summary,
        "clinical_numeric": clin_summary,
        "clinical_non_numeric_or_sparse_kept_as_is": clin_non_numeric,
        "feature_counts": {
            "expr_before": int(len(expr_cols)),
            "expr_after": int(len(expr_kept)),
            "clinical_before": int(len(clinical_cols)),
            "clinical_after": int(len(clinical_final)),
            "total_before": int(len(expr_cols) + len(clinical_cols)),
            "total_after": int(len(expr_kept) + len(clinical_final)),
            "total_dropped": int((len(expr_cols) + len(clinical_cols)) - (len(expr_kept) + len(clinical_final))),
        },
    }
    return clinical_final, expr_kept, dropped_df, summary
