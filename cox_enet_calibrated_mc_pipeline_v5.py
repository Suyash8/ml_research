
"""Cox Elastic-Net + calibrated horizon risk estimates + Monte Carlo survival summaries.

What this script does
- trains a Cox Elastic-Net survival model on genomic + clinical features
- uses a train / calibration / test split
- applies preprocessing, PCA for expression, and collinearity filtering
- reports C-index on train / calibration / test
- fits horizon-wise isotonic calibration models for 12 / 24 / 36 / 60 months
- reports horizon AUROC and Brier score on known-label test rows
- simulates survival times from the fitted Cox model + Breslow baseline hazard
- exports Monte Carlo uncertainty summaries (P10 / P50 / P90, horizon survival probs, RMST@60)

Important interpretation note
- The Monte Carlo summaries are simulation-based estimates from the fitted Cox model.
- They are not guarantees of individual survival.
- Exact survival month claims should not be made from this output.

This script intentionally does NOT include raw-month conformal prediction.
That layer is the hardest to justify under censoring and is omitted on purpose.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RANDOM_STATE = 42
EXPR_PCA_COMPONENTS = 50
MAXITER = 400
CV_FOLDS = 5

HORIZONS_MONTHS = [12.0, 24.0, 36.0, 60.0]
MC_RMST_HORIZON_MONTHS = 60.0
MC_N_SIMS = 5000
MC_RANDOM_STATE = RANDOM_STATE

ALPHA_GRID = [0.1, 0.3, 0.8, 1.5, 3.0]
L1_GRID = [0.0, 0.1, 0.3, 0.5, 0.7]
SMOOTH_L1_EPS = 1e-6

OUTLIER_COLUMNS = ("OS_MONTHS", "AGE")
OUTLIER_IQR_MULTIPLIER = 3.0

COLLINEARITY_THRESHOLD = 0.75
COLLINEARITY_MAX_FULL_CORR_FEATURES = 3000

BASE = Path("/home/illionar/Projects/ml_research")
INPUT_PATH = BASE / "data" / "preprocessed_cleaned" / "patient_multiomic_cleaned.parquet"
OUT_DIR = BASE / "data" / "model_outputs" / "cox_enet_calibrated_mc_outputs_v5"



# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


# ---------------------------------------------------------------------
# Data loading / cleaning
# ---------------------------------------------------------------------

def prepare_dataframe(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    df = pd.read_parquet(input_path).copy()

    required = ["OS_MONTHS", "OS_EVENT"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["OS_MONTHS"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    df["OS_EVENT"] = pd.to_numeric(df["OS_EVENT"], errors="coerce")
    df = df.dropna(subset=["OS_MONTHS", "OS_EVENT"]).copy()
    df = df[df["OS_MONTHS"] > 0].copy()
    df["OS_EVENT"] = (df["OS_EVENT"] > 0).astype(int)

    if "PATIENT_ID" not in df.columns:
        df["PATIENT_ID"] = [f"ROW_{i:07d}" for i in range(len(df))]
    df["PATIENT_ID"] = df["PATIENT_ID"].astype(str)

    if len(df) < 20:
        raise ValueError(f"Too few usable rows after cleaning: {len(df)}")

    n_events = int(df["OS_EVENT"].sum())
    n_cens = int(len(df) - n_events)
    if n_events == 0 or n_cens == 0:
        raise ValueError("Need both events and censored rows after cleaning.")

    print(f"Loaded {len(df)} rows | events={n_events} | censored={n_cens} | event_rate={n_events/len(df):.3f}")
    return df


def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: Sequence[str] = OUTLIER_COLUMNS,
    iqr_multiplier: float = OUTLIER_IQR_MULTIPLIER,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Optional helper retained for auditing. Not used by default in main()."""
    work = df.copy()
    report: Dict[str, Any] = {
        "initial_rows": int(len(work)),
        "iqr_multiplier": float(iqr_multiplier),
        "rules": [],
    }

    for col in columns:
        if col not in work.columns:
            report["rules"].append(
                {"column": col, "skipped": True, "reason": "column_not_found", "removed_rows": 0}
            )
            continue
        s = pd.to_numeric(work[col], errors="coerce")
        q1 = safe_float(s.quantile(0.25))
        q3 = safe_float(s.quantile(0.75))
        iqr = safe_float(q3 - q1)
        if not np.isfinite(iqr) or iqr <= 0:
            report["rules"].append(
                {
                    "column": col,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "skipped": True,
                    "reason": "non_positive_or_non_finite_iqr",
                    "removed_rows": 0,
                }
            )
            continue
        lo = float(q1 - iqr_multiplier * iqr)
        hi = float(q3 + iqr_multiplier * iqr)
        keep = s.isna() | ((s >= lo) & (s <= hi))
        removed = int((~keep).sum())
        report["rules"].append(
            {
                "column": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lo,
                "upper_bound": hi,
                "removed_rows": removed,
                "skipped": False,
            }
        )
        work = work.loc[keep].copy()
        if len(work) == 0:
            raise ValueError(f"Outlier filter removed all rows at column '{col}'.")

    report["final_rows"] = int(len(work))
    report["total_removed"] = int(report["initial_rows"] - report["final_rows"])
    return work, report


# ---------------------------------------------------------------------
# Splitting / feature selection
# ---------------------------------------------------------------------

def split_three_way(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Dict[str, Any]]]:
    idx = np.arange(len(df), dtype=int)
    y = df["OS_EVENT"].to_numpy(dtype=int)

    if len(np.unique(y)) < 2:
        raise ValueError("OS_EVENT has only one class; stratification is impossible.")

    idx_train, idx_tmp = train_test_split(
        idx, test_size=0.4, random_state=RANDOM_STATE, stratify=y
    )
    idx_cal, idx_test = train_test_split(
        idx_tmp, test_size=0.5, random_state=RANDOM_STATE, stratify=y[idx_tmp]
    )

    report: Dict[str, Dict[str, Any]] = {}
    for name, sidx in (("train", idx_train), ("calibration", idx_cal), ("test", idx_test)):
        nr = int(len(sidx))
        ne = int(y[sidx].sum())
        nc = nr - ne
        if ne == 0 or nc == 0:
            raise ValueError(f"Split '{name}' has only one class.")
        report[name] = {
            "rows": nr,
            "events": ne,
            "censored": nc,
            "event_rate": float(ne / nr),
        }
    return idx_train, idx_cal, idx_test, report


def get_feature_sets(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    clinical_candidates = ["AGE", "SEX", "RACE", "ETHNICITY", "CANCER_TYPE", "AGE_GROUP"]
    clinical_cols = [c for c in clinical_candidates if c in df.columns]
    expr_cols = sorted(c for c in df.columns if c.startswith("EXPR_"))
    if not clinical_cols and not expr_cols:
        raise ValueError("No usable feature columns found.")
    return clinical_cols, expr_cols


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


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

def _fit_clinical_block(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    clinical_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, Optional[ColumnTransformer], List[str], List[str]]:
    empty = lambda n: np.empty((n, 0), dtype=float)

    if not clinical_cols:
        return empty(len(df_tr)), empty(len(df_va)), None, [], []

    c_tr = df_tr[clinical_cols].copy()
    c_va = df_va[clinical_cols].copy()

    num_cols = c_tr.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in c_tr.columns if c not in num_cols]

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", _make_ohe())]), cat_cols))

    if not transformers:
        return empty(len(df_tr)), empty(len(df_va)), None, [], []

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    X_tr = np.asarray(pre.fit_transform(c_tr), dtype=float)
    X_va = np.asarray(pre.transform(c_va), dtype=float)
    return X_tr, X_va, pre, num_cols, cat_cols


def _fit_expression_block(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    expr_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, Optional[Pipeline], int]:
    empty = lambda n: np.empty((n, 0), dtype=float)

    if not expr_cols:
        return empty(len(df_tr)), empty(len(df_va)), None, 0

    e_tr = df_tr[expr_cols].copy()
    e_va = df_va[expr_cols].copy()
    n_comp = min(EXPR_PCA_COMPONENTS, int(e_tr.shape[0]), int(e_tr.shape[1]))
    if n_comp < 1:
        raise ValueError("Cannot configure PCA with fewer than 1 component.")

    pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("sc", StandardScaler()),
            ("pca", PCA(n_components=n_comp, random_state=RANDOM_STATE)),
        ]
    )
    X_tr = np.asarray(pipe.fit_transform(e_tr), dtype=float)
    X_va = np.asarray(pipe.transform(e_va), dtype=float)
    return X_tr, X_va, pipe, int(n_comp)


def fit_transform_features(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    clinical_cols: List[str],
    expr_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, Optional[ColumnTransformer], Optional[Pipeline], StandardScaler, Dict[str, Any]]:
    Xc_tr, Xc_va, clin_pre, num_cols, cat_cols = _fit_clinical_block(df_tr, df_va, clinical_cols)
    Xe_tr, Xe_va, expr_pipe, n_comp = _fit_expression_block(df_tr, df_va, expr_cols)

    blocks_tr = [a for a in (Xc_tr, Xe_tr) if a.shape[1] > 0]
    blocks_va = [a for a in (Xc_va, Xe_va) if a.shape[1] > 0]
    if not blocks_tr:
        raise ValueError("No usable features after preprocessing.")

    X_tr = np.hstack(blocks_tr)
    X_va = np.hstack(blocks_va)

    if X_tr.shape[1] != X_va.shape[1]:
        raise ValueError("Feature dimension mismatch between train and validation.")

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)

    if not (np.isfinite(X_tr).all() and np.isfinite(X_va).all()):
        raise ValueError("Non-finite values after scaling.")

    stds = np.nanstd(X_tr, axis=0)
    checks: Dict[str, Any] = {
        "clinical_num_cols": num_cols,
        "clinical_cat_cols": cat_cols,
        "clinical_output_dim": int(Xc_tr.shape[1]),
        "expr_input_dim": int(len(expr_cols)),
        "expr_pca_components": int(n_comp),
        "expr_output_dim": int(Xe_tr.shape[1]),
        "final_dim": int(X_tr.shape[1]),
        "final_train_mean_abs_max": float(np.max(np.abs(np.nanmean(X_tr, axis=0)))),
        "final_train_std_min": float(np.min(stds)),
        "final_train_std_max": float(np.max(stds)),
        "final_constant_feature_count": int(np.sum(stds < 1e-12)),
    }
    return X_tr, X_va, clin_pre, expr_pipe, scaler, checks


def transform_features(
    df: pd.DataFrame,
    clinical_cols: List[str],
    expr_cols: List[str],
    clin_pre: Optional[ColumnTransformer],
    expr_pipe: Optional[Pipeline],
    scaler: StandardScaler,
) -> np.ndarray:
    blocks = []
    if clinical_cols:
        if clin_pre is None:
            raise ValueError("Clinical preprocessor missing.")
        blocks.append(np.asarray(clin_pre.transform(df[clinical_cols].copy()), dtype=float))
    if expr_cols:
        if expr_pipe is None:
            raise ValueError("Expression preprocessor missing.")
        blocks.append(np.asarray(expr_pipe.transform(df[expr_cols].copy()), dtype=float))
    if not blocks:
        raise ValueError("No feature blocks to transform.")
    X = scaler.transform(np.hstack(blocks))
    if not np.isfinite(X).all():
        raise ValueError("Non-finite values after transform.")
    return X


# ---------------------------------------------------------------------
# Cox Elastic-Net
# ---------------------------------------------------------------------

class CoxElasticNet:
    """Cox proportional hazards model with elastic-net regularisation."""

    def __init__(
        self,
        alpha: float = 0.8,
        l1_ratio: float = 0.3,
        smooth_l1_eps: float = SMOOTH_L1_EPS,
        maxiter: int = MAXITER,
    ):
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.smooth_l1_eps = float(smooth_l1_eps)
        self.maxiter = int(maxiter)

    def _nll_grad(
        self,
        beta: np.ndarray,
        X: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        # Proper Breslow-tied Cox partial likelihood.
        order = np.argsort(time, kind="mergesort")
        Xo = np.asarray(X[order], dtype=float)
        to = np.asarray(time[order], dtype=float)
        eo = np.asarray(event[order], dtype=int)

        eta = np.clip(Xo @ beta, -40, 40)
        exp_eta = np.exp(eta)

        # Suffix sums for risk sets at each time point.
        s0_suffix = np.cumsum(exp_eta[::-1])[::-1]
        s1_suffix = np.cumsum((exp_eta[:, None] * Xo)[::-1], axis=0)[::-1]

        event_times = np.unique(to[eo == 1])
        loglik = 0.0
        grad_loglik = np.zeros_like(beta, dtype=float)

        for t in event_times:
            idx0 = int(np.searchsorted(to, t, side="left"))
            at_risk_s0 = float(s0_suffix[idx0])
            if not np.isfinite(at_risk_s0) or at_risk_s0 <= 0:
                continue

            mask_t = to == t
            d = int(np.sum(eo[mask_t] == 1))
            if d <= 0:
                continue

            event_mask = mask_t & (eo == 1)
            eta_events = eta[event_mask]
            X_events = Xo[event_mask]

            loglik += float(np.sum(eta_events) - d * np.log(at_risk_s0))
            grad_loglik += np.sum(X_events, axis=0) - d * (s1_suffix[idx0] / at_risk_s0)

        # Smooth elastic-net penalty.
        l1 = np.sum(np.sqrt(beta * beta + self.smooth_l1_eps))
        l2 = 0.5 * float(np.dot(beta, beta))
        grad_l1 = beta / np.sqrt(beta * beta + self.smooth_l1_eps)

        nll = -float(loglik) + self.alpha * (self.l1_ratio * l1 + (1.0 - self.l1_ratio) * l2)
        grad = -grad_loglik + self.alpha * (self.l1_ratio * grad_l1 + (1.0 - self.l1_ratio) * beta)
        return float(nll), np.asarray(grad, dtype=float)

    def fit(self, X: np.ndarray, time: np.ndarray, event: np.ndarray) -> "CoxElasticNet":
        beta0 = np.zeros(X.shape[1], dtype=float)
        res = minimize(
            fun=lambda b: self._nll_grad(b, X, time, event)[0],
            x0=beta0,
            jac=lambda b: self._nll_grad(b, X, time, event)[1],
            method="L-BFGS-B",
            options={"maxiter": self.maxiter},
        )
        self.coef_ = np.asarray(res.x, dtype=float)
        self.success_ = bool(res.success)
        self.n_iter_ = int(res.nit)
        self.message_ = str(res.message)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_


def concordance_index_censored(times: np.ndarray, events: np.ndarray, risk_scores: np.ndarray) -> float:
    """Harrell-style C-index, higher risk means shorter survival."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    risk_scores = np.asarray(risk_scores, dtype=float)

    t_i = times[:, None]
    t_j = times[None, :]
    e_i = events[:, None]
    comparable = (t_i < t_j) & (e_i == 1)
    denom = int(comparable.sum())
    if denom == 0:
        return float("nan")

    s_i = risk_scores[:, None]
    s_j = risk_scores[None, :]
    concordant = int(((s_i > s_j) & comparable).sum())
    tied = int(((s_i == s_j) & comparable).sum())
    return float((concordant + 0.5 * tied) / denom)


# ---------------------------------------------------------------------
# Cross-validation tuning
# ---------------------------------------------------------------------

def run_cv_tuning(df_train: pd.DataFrame, clinical_cols: List[str], expr_cols: List[str]) -> pd.DataFrame:
    y = df_train["OS_EVENT"].to_numpy(dtype=int)
    class_counts = np.bincount(y, minlength=2)
    positive_counts = class_counts[class_counts > 0]
    if len(positive_counts) < 2:
        raise ValueError("CV requires both event classes.")
    minority = int(np.min(positive_counts))
    n_splits = min(CV_FOLDS, minority)
    if n_splits < 2:
        raise ValueError(f"Too few minority samples ({minority}) for CV.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for cox_alpha in ALPHA_GRID:
        for l1_ratio in L1_GRID:
            fold_scores: List[float] = []
            failures: List[str] = []

            for fold_id, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(df_train)), y), start=1):
                tr = df_train.iloc[tr_idx]
                va = df_train.iloc[va_idx]
                try:
                    X_tr, X_va, _, _, _, _ = fit_transform_features(tr, va, clinical_cols, expr_cols)
                    mdl = CoxElasticNet(alpha=cox_alpha, l1_ratio=l1_ratio, maxiter=MAXITER)
                    mdl.fit(X_tr, tr["OS_MONTHS"].to_numpy(float), tr["OS_EVENT"].to_numpy(int))
                    if not bool(getattr(mdl, "success_", True)):
                        failures.append(f"fold_{fold_id}:not_converged")
                        continue
                    ci = concordance_index_censored(
                        va["OS_MONTHS"].to_numpy(float),
                        va["OS_EVENT"].to_numpy(int),
                        mdl.predict_risk(X_va),
                    )
                    if not np.isfinite(ci):
                        failures.append(f"fold_{fold_id}:non_finite_ci")
                        continue
                    fold_scores.append(float(ci))
                except Exception as exc:
                    failures.append(f"fold_{fold_id}:{type(exc).__name__}")

            rows.append(
                {
                    "alpha": float(cox_alpha),
                    "l1_ratio": float(l1_ratio),
                    "n_valid_folds": int(len(fold_scores)),
                    "n_failed_folds": int(n_splits - len(fold_scores)),
                    "mean_c_index": float(np.mean(fold_scores)) if fold_scores else float("nan"),
                    "std_c_index": float(np.std(fold_scores)) if fold_scores else float("nan"),
                    "fold_scores": ", ".join(f"{s:.4f}" for s in fold_scores),
                    "failure_notes": "; ".join(failures),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["n_valid_folds", "mean_c_index", "std_c_index"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


def select_best_hyperparameters(cv_df: pd.DataFrame) -> pd.Series:
    viable = cv_df[np.isfinite(cv_df["mean_c_index"]) & (cv_df["n_valid_folds"] >= 2)].copy()
    if viable.empty:
        raise ValueError("No viable hyperparameter combination in CV.")
    return viable.sort_values(
        ["n_valid_folds", "mean_c_index", "std_c_index"], ascending=[False, False, True]
    ).iloc[0]


# ---------------------------------------------------------------------
# Horizon calibration and metrics
# ---------------------------------------------------------------------

def _horizon_labels(time_arr: np.ndarray, event_arr: np.ndarray, horizon: float) -> Tuple[np.ndarray, np.ndarray]:
    """Keep rows whose label is known at this horizon."""
    confirmed_event = (event_arr == 1) & (time_arr <= horizon)
    confirmed_no_event = time_arr > horizon
    keep = confirmed_event | confirmed_no_event
    y = np.where(confirmed_event[keep], 1, 0).astype(int)
    return keep, y


def fit_horizon_calibrators(
    r_cal: np.ndarray,
    t_cal: np.ndarray,
    e_cal: np.ndarray,
    horizons: Sequence[float] = HORIZONS_MONTHS,
) -> Dict[float, Dict[str, Any]]:
    calibrators: Dict[float, Dict[str, Any]] = {}

    for h in horizons:
        keep_cal, y_cal = _horizon_labels(t_cal, e_cal, float(h))
        n_cal = int(len(y_cal))
        base: Dict[str, Any] = {"horizon_months": float(h), "n_cal_known": n_cal}

        if n_cal == 0 or len(np.unique(y_cal)) < 2:
            calibrators[float(h)] = {**base, "status": "skipped:insufficient_calibration_labels"}
            continue

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(r_cal[keep_cal], y_cal)
        p_cal = iso.predict(r_cal[keep_cal])

        calibrators[float(h)] = {
            **base,
            "status": "ok",
            "isotonic": iso,
            "cal_event_rate": float(np.mean(y_cal)),
            "cal_brier": float(brier_score_loss(y_cal, p_cal)),
            "cal_auc": float(roc_auc_score(y_cal, p_cal)) if len(np.unique(y_cal)) == 2 else float("nan"),
        }

    return calibrators


def horizon_test_metrics(
    r_te: np.ndarray,
    t_te: np.ndarray,
    e_te: np.ndarray,
    calibrators: Dict[float, Dict[str, Any]],
    horizons: Sequence[float] = HORIZONS_MONTHS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []

    for h in horizons:
        h = float(h)
        keep_te, y_te = _horizon_labels(t_te, e_te, h)
        n_known = int(len(y_te))
        cal_info = calibrators.get(h, {})
        status = cal_info.get("status", "missing")

        if n_known == 0 or len(np.unique(y_te)) < 2 or status != "ok":
            metric_rows.append(
                {
                    "horizon_months": h,
                    "n_test_known": n_known,
                    "event_rate_test_known": float(np.mean(y_te)) if n_known else float("nan"),
                    "auc": float("nan"),
                    "brier": float("nan"),
                    "note": f"skipped:{status}",
                }
            )
            continue

        iso: IsotonicRegression = cal_info["isotonic"]
        p_event = iso.predict(r_te[keep_te])
        p_survive = 1.0 - p_event

        try:
            auc = float(roc_auc_score(y_te, p_event))
        except Exception:
            auc = float("nan")
        try:
            brier = float(brier_score_loss(y_te, p_event))
        except Exception:
            brier = float("nan")

        metric_rows.append(
            {
                "horizon_months": h,
                "n_test_known": n_known,
                "event_rate_test_known": float(np.mean(y_te)),
                "auc": auc,
                "brier": brier,
                "note": "ok",
            }
        )

        known_idx = np.where(keep_te)[0]
        for i_local, i_global in enumerate(known_idx):
            pred_rows.append(
                {
                    "horizon_months": h,
                    "test_row_index_within_split": int(i_global),
                    "risk_score": float(r_te[i_global]),
                    "y_true_by_horizon": int(y_te[i_local]),
                    "p_event_hat": float(p_event[i_local]),
                    "p_survive_hat": float(p_survive[i_local]),
                }
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows)


# ---------------------------------------------------------------------
# Baseline hazard / Monte Carlo
# ---------------------------------------------------------------------

def fit_breslow_baseline_hazard(
    times: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    risk_scores = np.asarray(risk_scores, dtype=float)

    if len(times) == 0:
        return np.array([], dtype=float), np.array([], dtype=float), {"status": "skipped:empty_training_data"}

    order = np.argsort(times, kind="mergesort")
    times = times[order]
    events = events[order]
    risk_scores = risk_scores[order]

    exp_eta = np.exp(np.clip(risk_scores, -40, 40))
    risk_suffix = np.cumsum(exp_eta[::-1])[::-1]

    event_times = np.unique(times[events == 1])
    base_times: List[float] = []
    base_cumhaz: List[float] = []
    cumhaz = 0.0

    for t in event_times:
        first_idx = int(np.searchsorted(times, t, side="left"))
        at_risk = float(risk_suffix[first_idx])
        d = int(np.sum((times == t) & (events == 1)))
        if at_risk <= 0 or d <= 0:
            continue
        cumhaz += float(d / at_risk)
        base_times.append(float(t))
        base_cumhaz.append(float(cumhaz))

    meta = {
        "status": "ok" if base_times else "skipped:no_event_times",
        "n_rows": int(len(times)),
        "n_events": int(events.sum()),
        "n_unique_event_times": int(len(base_times)),
        "max_observed_followup_months": float(np.max(times)),
        "max_event_time_months": float(base_times[-1]) if base_times else float("nan"),
    }
    return np.asarray(base_times, dtype=float), np.asarray(base_cumhaz, dtype=float), meta


def simulate_cox_survival_times(
    risk_scores: np.ndarray,
    baseline_times: np.ndarray,
    baseline_cumhaz: np.ndarray,
    max_followup_months: float,
    n_sims: int = MC_N_SIMS,
    random_state: int = MC_RANDOM_STATE,
) -> np.ndarray:
    risk_scores = np.asarray(risk_scores, dtype=float)
    baseline_times = np.asarray(baseline_times, dtype=float)
    baseline_cumhaz = np.asarray(baseline_cumhaz, dtype=float)

    n_patients = int(len(risk_scores))
    if n_patients == 0:
        return np.empty((0, int(n_sims)), dtype=float)
    if len(baseline_times) == 0 or len(baseline_cumhaz) == 0:
        return np.full((n_patients, int(n_sims)), np.nan, dtype=float)

    rng = np.random.default_rng(random_state)
    u = np.clip(rng.uniform(size=(n_patients, int(n_sims))), 1e-12, 1.0 - 1e-12)

    eta = np.clip(risk_scores[:, None], -40, 40)
    target_cumhaz = -np.log(u) / np.exp(eta)

    idx = np.searchsorted(baseline_cumhaz, target_cumhaz, side="left")
    sim_times = np.full_like(target_cumhaz, float(max_followup_months), dtype=float)

    valid = idx < len(baseline_times)
    sim_times[valid] = baseline_times[idx[valid]]
    return sim_times


def monte_carlo_survival_block(
    t_tr: np.ndarray,
    e_tr: np.ndarray,
    r_tr: np.ndarray,
    t_te: np.ndarray,
    e_te: np.ndarray,
    r_te: np.ndarray,
    patient_ids_te: Optional[np.ndarray] = None,
    n_sims: int = MC_N_SIMS,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    baseline_times, baseline_cumhaz, baseline_meta = fit_breslow_baseline_hazard(t_tr, e_tr, r_tr)
    sim_times = simulate_cox_survival_times(
        r_te,
        baseline_times=baseline_times,
        baseline_cumhaz=baseline_cumhaz,
        max_followup_months=baseline_meta.get("max_observed_followup_months", float(np.nan)),
        n_sims=n_sims,
        random_state=MC_RANDOM_STATE,
    )

    if sim_times.size == 0:
        empty = pd.DataFrame(columns=["test_row_index_within_split", "PATIENT_ID"])
        return empty, baseline_meta, {"status": "skipped:empty_test_set"}

    rows = []
    for i in range(sim_times.shape[0]):
        s = sim_times[i]
        rows.append(
            {
                "test_row_index_within_split": int(i),
                "PATIENT_ID": str(patient_ids_te[i]) if patient_ids_te is not None else f"TEST_{i:07d}",
                "mc_survival_p10_months": float(np.quantile(s, 0.10)),
                "mc_survival_p50_months": float(np.quantile(s, 0.50)),
                "mc_survival_p90_months": float(np.quantile(s, 0.90)),
                "mc_prob_survive_12_months": float(np.mean(s >= 12.0)),
                "mc_prob_survive_24_months": float(np.mean(s >= 24.0)),
                "mc_prob_survive_36_months": float(np.mean(s >= 36.0)),
                "mc_prob_survive_60_months": float(np.mean(s >= 60.0)),
                "mc_rmst_60_months": float(np.mean(np.minimum(s, MC_RMST_HORIZON_MONTHS))),
                "mc_note": "simulation_based_summary_from_fitted_cox_model",
            }
        )

    baseline_meta = {**baseline_meta, "baseline_times": baseline_times, "baseline_cumhaz": baseline_cumhaz}
    summary = {
        "status": "ok",
        "n_test": int(sim_times.shape[0]),
        "n_sims": int(n_sims),
        "probability_horizons_months": list(HORIZONS_MONTHS),
        "rmst_horizon_months": float(MC_RMST_HORIZON_MONTHS),
        "interpretation": "bounded Monte Carlo summaries from the fitted Cox model; p90 is an upper-plausible survival time, not a guarantee",
    }
    return pd.DataFrame(rows), baseline_meta, summary


def survival_probabilities_from_breslow(
    risk_scores: np.ndarray,
    baseline_times: np.ndarray,
    baseline_cumhaz: np.ndarray,
    horizons: Sequence[float] = HORIZONS_MONTHS,
) -> pd.DataFrame:
    """Deterministic survival probability at fixed horizons from the fitted Cox model."""
    risk_scores = np.asarray(risk_scores, dtype=float)
    out = pd.DataFrame({"risk_score": risk_scores})

    for h in horizons:
        h = float(h)
        if len(baseline_times) == 0:
            out[f"cox_survival_prob_{int(h)}m"] = np.nan
            continue
        idx = int(np.searchsorted(baseline_times, h, side="right") - 1)
        base_h = 0.0 if idx < 0 else float(baseline_cumhaz[idx])
        out[f"cox_survival_prob_{int(h)}m"] = np.exp(-base_h * np.exp(np.clip(risk_scores, -40, 40)))
    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Cox ENet survival pipeline with calibration and Monte Carlo summaries.")
    parser.add_argument("--input", type=Path, default=INPUT_PATH, help="Input parquet file.")
    parser.add_argument("--outdir", type=Path, default=OUT_DIR, help="Output directory.")
    parser.add_argument("--no-outlier-filter", action="store_true", help="Keep all rows; default is to keep all rows anyway.")
    parser.add_argument("--apply-outlier-filter", action="store_true", help="Optionally apply IQR filtering on OS_MONTHS and AGE.")
    args = parser.parse_args()

    input_path: Path = args.input
    out_dir: Path = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_dataframe(input_path).reset_index(drop=True)

    if args.apply_outlier_filter and not args.no_outlier_filter:
        df, outlier_report = remove_outliers_iqr(df)
        print(f"After outlier filter: {len(df)} rows")
    else:
        outlier_report = {
            "applied": False,
            "reason": "not_used_by_default_to_avoid_target-informed filtering",
            "rows": int(len(df)),
        }

    idx_train, idx_cal, idx_test, split_report = split_three_way(df)

    clinical_cols, expr_cols = get_feature_sets(df)
    print(f"Clinical features: {len(clinical_cols)} | Expression features: {len(expr_cols)}")

    df_train_for_filter = df.iloc[idx_train].copy().reset_index(drop=True)
    clinical_cols_f, expr_cols_f, dropped_df, coll_summary = apply_collinearity_filter(
        df_train_for_filter,
        clinical_cols,
        expr_cols,
        threshold=COLLINEARITY_THRESHOLD,
    )

    dropped_df.to_csv(out_dir / "collinearity_dropped_features.csv", index=False)
    save_json(out_dir / "collinearity_summary.json", coll_summary)

    print(f"After collinearity: clinical {len(clinical_cols)} -> {len(clinical_cols_f)} | expr {len(expr_cols)} -> {len(expr_cols_f)}")

    if not clinical_cols_f and not expr_cols_f:
        raise ValueError("All features removed by collinearity filtering.")

    selected = set(clinical_cols_f) | set(expr_cols_f)
    forbidden_exact = {"OS_MONTHS", "OS_EVENT", "OS_STATUS", "DFS_STATUS", "DFS_EVENT"}
    forbidden_prefixes = ("OS_", "DFS_")
    leaking = sorted(
        c for c in selected if c in forbidden_exact or any(c.startswith(p) for p in forbidden_prefixes)
    )
    if leaking:
        raise ValueError(f"Target-leakage features detected: {leaking}")

    # Split dataframes
    tr = df.iloc[idx_train].copy().reset_index(drop=True)
    cal = df.iloc[idx_cal].copy().reset_index(drop=True)
    te = df.iloc[idx_test].copy().reset_index(drop=True)

    df_train = tr.copy()

    print("Running CV hyperparameter search ...")
    cv_df = run_cv_tuning(df_train, clinical_cols_f, expr_cols_f)
    cv_df.to_csv(out_dir / "hyperparameter_cv_results.csv", index=False)
    best = select_best_hyperparameters(cv_df)
    best_cox_alpha = float(best["alpha"])
    best_l1 = float(best["l1_ratio"])
    print(
        f"Best params: alpha={best_cox_alpha}, l1_ratio={best_l1}, CV c-index={float(best['mean_c_index']):.4f}"
    )

    X_tr, X_cal, clin_pre, expr_pipe, scaler, feat_checks = fit_transform_features(
        tr, cal, clinical_cols_f, expr_cols_f
    )
    X_te = transform_features(te, clinical_cols_f, expr_cols_f, clin_pre, expr_pipe, scaler)

    t_tr = tr["OS_MONTHS"].to_numpy(float)
    e_tr = tr["OS_EVENT"].to_numpy(int)
    t_cal = cal["OS_MONTHS"].to_numpy(float)
    e_cal = cal["OS_EVENT"].to_numpy(int)
    t_te = te["OS_MONTHS"].to_numpy(float)
    e_te = te["OS_EVENT"].to_numpy(int)

    print("Fitting final Cox model ...")
    model = CoxElasticNet(alpha=best_cox_alpha, l1_ratio=best_l1, maxiter=MAXITER)
    model.fit(X_tr, t_tr, e_tr)
    if not bool(getattr(model, "success_", True)):
        raise RuntimeError(f"Cox optimizer did not converge: {getattr(model, 'message_', 'unknown')}")

    r_tr = model.predict_risk(X_tr)
    r_cal = model.predict_risk(X_cal)
    r_te = model.predict_risk(X_te)

    ci_train = float(concordance_index_censored(t_tr, e_tr, r_tr))
    ci_cal = float(concordance_index_censored(t_cal, e_cal, r_cal))
    ci_test = float(concordance_index_censored(t_te, e_te, r_te))
    print(f"C-index: train={ci_train:.4f} | cal={ci_cal:.4f} | test={ci_test:.4f}")

    print("Fitting horizon calibrators ...")
    calibrators = fit_horizon_calibrators(r_cal, t_cal, e_cal, horizons=HORIZONS_MONTHS)
    horizon_metrics, horizon_preds = horizon_test_metrics(
        r_te, t_te, e_te, calibrators, horizons=HORIZONS_MONTHS
    )
    horizon_metrics.to_csv(out_dir / "time_dependent_horizon_metrics.csv", index=False)
    horizon_preds.to_csv(out_dir / "time_dependent_horizon_predictions.csv", index=False)

    print("Running Monte Carlo survival summaries ...")
    mc_te, mc_baseline_meta, mc_summary = monte_carlo_survival_block(
        t_tr,
        e_tr,
        r_tr,
        t_te,
        e_te,
        r_te,
        patient_ids_te=te["PATIENT_ID"].astype(str).to_numpy(),
        n_sims=MC_N_SIMS,
    )
    mc_te.to_csv(out_dir / "monte_carlo_survival_test_predictions.csv", index=False)

    # Deterministic Cox survival probabilities at horizons (useful as an additional check).
    cox_prob_te = survival_probabilities_from_breslow(
        r_te,
        baseline_times=np.asarray(mc_baseline_meta.get("baseline_times", np.array([])), dtype=float),
        baseline_cumhaz=np.asarray(mc_baseline_meta.get("baseline_cumhaz", np.array([])), dtype=float),
        horizons=HORIZONS_MONTHS,
    )

    # Main per-row output for train/cal/test.
    patient_ids = pd.concat([tr["PATIENT_ID"], cal["PATIENT_ID"], te["PATIENT_ID"]], axis=0).astype(str).values
    pred_df = pd.DataFrame(
        {
            "split": (["train"] * len(tr) + ["calibration"] * len(cal) + ["test"] * len(te)),
            "PATIENT_ID": patient_ids,
            "OS_MONTHS": np.concatenate([t_tr, t_cal, t_te]),
            "OS_EVENT": np.concatenate([e_tr, e_cal, e_te]),
            "risk_score": np.concatenate([r_tr, r_cal, r_te]),
        }
    )

    # Add calibrated horizon probabilities on the full dataset where possible.
    for h in HORIZONS_MONTHS:
        h = float(h)
        cal_info = calibrators.get(h, {})
        if cal_info.get("status") == "ok":
            iso: IsotonicRegression = cal_info["isotonic"]
            p_event_all = iso.predict(np.concatenate([r_tr, r_cal, r_te]))
            pred_df[f"cal_event_prob_{int(h)}m"] = p_event_all
            pred_df[f"cal_survival_prob_{int(h)}m"] = 1.0 - p_event_all
        else:
            pred_df[f"cal_event_prob_{int(h)}m"] = np.nan
            pred_df[f"cal_survival_prob_{int(h)}m"] = np.nan

    # Add deterministic Cox survival probabilities for the test set only.
    for col in cox_prob_te.columns:
        if col == "risk_score":
            continue
        pred_df[f"test_{col}"] = np.nan
    test_start = len(tr) + len(cal)
    for idx_col in [f"cox_survival_prob_{int(float(h))}m" for h in HORIZONS_MONTHS]:
        pred_df.loc[test_start:test_start + len(te) - 1, f"test_{idx_col}"] = cox_prob_te[idx_col].to_numpy()

    # Add Monte Carlo summaries for test rows only.
    for col in [
        "mc_survival_p10_months",
        "mc_survival_p50_months",
        "mc_survival_p90_months",
        "mc_prob_survive_12_months",
        "mc_prob_survive_24_months",
        "mc_prob_survive_36_months",
        "mc_prob_survive_60_months",
        "mc_rmst_60_months",
    ]:
        pred_df[col] = np.nan
    pred_df.loc[test_start:test_start + len(te) - 1, mc_te.columns.intersection(pred_df.columns)] = mc_te[
        mc_te.columns.intersection(pred_df.columns)
    ].to_numpy()

    pred_df.to_csv(out_dir / "main_predictions.csv", index=False)
    pred_df.to_csv(out_dir / "tuned_model_predictions.csv", index=False)

    coef_df = pd.DataFrame(
        {"coef_index": np.arange(len(model.coef_)), "coef_value": model.coef_}
    ).sort_values("coef_value", key=np.abs, ascending=False)
    coef_df.to_csv(out_dir / "coefficient_exports.csv", index=False)
    coef_df.to_csv(out_dir / "tuned_model_coefficients.csv", index=False)

    mc_widths = (mc_te["mc_survival_p90_months"] - mc_te["mc_survival_p10_months"]).to_numpy(dtype=float)
    mc_widths = mc_widths[np.isfinite(mc_widths)]
    mc_mean_width = float(np.mean(mc_widths)) if len(mc_widths) else float("nan")
    mc_median_width = float(np.median(mc_widths)) if len(mc_widths) else float("nan")
    mc_p90_width = float(np.percentile(mc_widths, 90)) if len(mc_widths) else float("nan")

    metrics: Dict[str, Any] = {
        "input_file": str(input_path),
        "best_cox_params": {"alpha": best_cox_alpha, "l1_ratio": best_l1},
        "cv_best_mean_c_index": float(best["mean_c_index"]),
        "cv_best_std_c_index": float(best["std_c_index"]),
        "cv_best_valid_folds": int(best["n_valid_folds"]),
        "n_total_rows": int(len(df)),
        "n_train": int(len(tr)),
        "n_calibration": int(len(cal)),
        "n_test": int(len(te)),
        "events_train": int(e_tr.sum()),
        "events_calibration": int(e_cal.sum()),
        "events_test": int(e_te.sum()),
        "cox_optimizer_success": bool(getattr(model, "success_", True)),
        "cox_optimizer_message": str(getattr(model, "message_", "unknown")),
        "cox_optimizer_iterations": int(getattr(model, "n_iter_", -1)),
        "c_index_train": ci_train,
        "c_index_calibration": ci_cal,
        "c_index_test": ci_test,
        "collinearity_filter": {
            "threshold_abs_pearson": COLLINEARITY_THRESHOLD,
            "clinical_before": int(len(clinical_cols)),
            "clinical_after": int(len(clinical_cols_f)),
            "expr_before": int(len(expr_cols)),
            "expr_after": int(len(expr_cols_f)),
        },
        "horizon_metrics": horizon_metrics.to_dict(orient="records"),
        "monte_carlo": {
            **mc_summary,
            "mean_interval_width_months": mc_mean_width,
            "median_interval_width_months": mc_median_width,
            "p90_interval_width_months": mc_p90_width,
        },
    }

    save_json(out_dir / "metrics.json", metrics)
    save_json(out_dir / "tuned_model_metrics.json", metrics)

    consistency: Dict[str, Any] = {
        "input_file": str(input_path),
        "rows_before_outlier_filter": int(len(df)),
        "outlier_report": outlier_report,
        "split_report": split_report,
        "feature_checks": feat_checks,
        "feature_counts": {
            "clinical_before_collinearity": int(len(clinical_cols)),
            "expr_before_collinearity": int(len(expr_cols)),
            "clinical_after_collinearity": int(len(clinical_cols_f)),
            "expr_after_collinearity": int(len(expr_cols_f)),
        },
        "collinearity_summary": coll_summary,
        "leakage_check_passed": True,
        "monte_carlo_baseline_meta": {
            k: v for k, v in mc_baseline_meta.items() if k not in {"baseline_times", "baseline_cumhaz"}
        },
    }
    save_json(out_dir / "consistency_checks.json", consistency)
    save_json(out_dir / "audit.json", consistency)

    artifact: Dict[str, Any] = {
        "clinical_cols_before_collinearity": clinical_cols,
        "expr_cols_before_collinearity": expr_cols,
        "clinical_cols_after_collinearity": clinical_cols_f,
        "expr_cols_after_collinearity": expr_cols_f,
        "clin_pre": clin_pre,
        "expr_pipe": expr_pipe,
        "scaler": scaler,
        "cox_model": model,
        "best_cox_alpha": best_cox_alpha,
        "best_cox_l1_ratio": best_l1,
        "horizon_calibrators": calibrators,
        "horizons_months": HORIZONS_MONTHS,
        "mc_summary": mc_summary,
        "mc_baseline_meta": {
            k: v for k, v in mc_baseline_meta.items() if k not in {"baseline_times", "baseline_cumhaz"}
        },
    }
    with open(out_dir / "final_locked_model.pkl", "wb") as f:
        pickle.dump(artifact, f)

    readme = [
        "# Cox ENet + calibrated horizon risk estimation + Monte Carlo",
        "",
        "## What this pipeline reports",
        "- risk score from a fitted Cox Elastic-Net model",
        "- horizon-wise calibrated event and survival probabilities at 12 / 24 / 36 / 60 months",
        "- C-index on train / calibration / test",
        "- horizon AUROC and Brier score on known-label test rows",
        "- Monte Carlo survival summaries from the fitted Cox model and Breslow baseline hazard",
        "- bounded RMST at 60 months",
        "",
        "## Important interpretation note",
        "The Monte Carlo block does not predict an exact survival lifespan.",
        "It gives simulation-based summaries under the fitted Cox model and",
        "the estimated Breslow baseline hazard.",
        "The p90 simulated survival month is an upper-plausible survival time, not a guarantee.",
        "",
        "## Files",
        "- main_predictions.csv",
        "- monte_carlo_survival_test_predictions.csv",
        "- time_dependent_horizon_metrics.csv",
        "- time_dependent_horizon_predictions.csv",
        "- tuned_model_predictions.csv",
        "- tuned_model_coefficients.csv",
        "- tuned_model_metrics.json",
        "- metrics.json",
        "- consistency_checks.json",
        "- audit.json",
        "- final_locked_model.pkl",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print("\nDone.")
    print(f"Results -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
