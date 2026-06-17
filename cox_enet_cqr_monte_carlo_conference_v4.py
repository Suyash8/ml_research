"""Cox Elastic-Net + raw-month CQR + Monte Carlo survival summaries.

This version keeps the original pipeline logic intact:
- Cox Elastic-Net core model
- train / calibration / test split
- PCA + preprocessing + collinearity filtering
- CV hyperparameter tuning
- raw-month conformalized quantile regression intervals
- time-dependent horizon classification diagnostics

Added layer:
- Monte Carlo survival summaries derived from the fitted Cox model and the
  Breslow baseline hazard estimate

Reporting discipline:
- no exact survival lifespan claims
- no guaranteed survival claims
- Monte Carlo outputs are presented only as bounded probabilistic summaries
  and bounded RMST estimates
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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


# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
EXPR_PCA_COMPONENTS = 50
MAXITER = 400
CV_FOLDS = 5
CONFORMAL_ALPHA = 0.10
HORIZONS_MONTHS = [12, 24, 36, 60]

ALPHA_GRID = [0.1, 0.3, 0.8, 1.5, 3.0]
L1_GRID = [0.0, 0.1, 0.3, 0.5, 0.7]
SMOOTH_L1_EPS = 1e-6

OUTLIER_IQR_MULTIPLIER = 3.0
OUTLIER_COLUMNS = ("OS_MONTHS", "AGE")
COLLINEARITY_THRESHOLD = 0.75
COLLINEARITY_MAX_FULL_CORR_FEATURES = 3000

CQR_MIN_TRAIN_EVENTS = 20
CQR_MIN_CAL_EVENTS = 10
CQR_GBR_PARAMS: Dict[str, Any] = {
    "loss": "quantile",
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.04,
    "min_samples_leaf": 10,
    "min_samples_split": 10,
    "subsample": 0.8,
    "random_state": RANDOM_STATE,
}
MAX_INTERVAL_WIDTH_MONTHS: Optional[float] = None

MC_N_SIMS = 5000
MC_RANDOM_STATE = RANDOM_STATE
MC_HORIZONS_MONTHS = [12, 24, 36, 60]
MC_RMST_HORIZON_MONTHS = 60.0

BASE = Path("/home/illionar/Projects/ml_research")
INPUT_PATH = BASE / "data" / "preprocessed_cleaned" / "patient_multiomic_cleaned.parquet"
OUT_DIR = BASE / "data" / "model_outputs" / "cox_enet_cqr_fixed_mc_v4"


# ---------------------------------------------------------------------------
# Cox Elastic-Net model
# ---------------------------------------------------------------------------

class CoxElasticNet:
    """Cox proportional hazards model with elastic-net regularisation."""

    def __init__(self, alpha: float = 0.8, l1_ratio: float = 0.3, smooth_l1_eps: float = SMOOTH_L1_EPS, maxiter: int = MAXITER):
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.smooth_l1_eps = float(smooth_l1_eps)
        self.maxiter = int(maxiter)

    def _nll_grad(self, beta: np.ndarray, X: np.ndarray, time: np.ndarray, event: np.ndarray) -> Tuple[float, np.ndarray]:
        order = np.argsort(-time)
        Xo, eo = X[order], event[order]

        eta = np.clip(Xo @ beta, -40, 40)
        exp_eta = np.exp(eta)
        s0 = np.cumsum(exp_eta)
        s1 = np.cumsum(exp_eta[:, None] * Xo, axis=0)

        idx = np.where(eo == 1)[0]
        l1 = np.sqrt(beta * beta + self.smooth_l1_eps).sum()
        l2 = 0.5 * np.dot(beta, beta)
        grad_l1 = beta / np.sqrt(beta * beta + self.smooth_l1_eps)

        if len(idx) == 0:
            nll = self.alpha * (self.l1_ratio * l1 + (1.0 - self.l1_ratio) * l2)
            grad = self.alpha * (self.l1_ratio * grad_l1 + (1.0 - self.l1_ratio) * beta)
            return float(nll), grad

        loglik = np.sum(eta[idx] - np.log(s0[idx]))
        grad_loglik = np.sum(Xo[idx] - s1[idx] / s0[idx, None], axis=0)
        nll = -float(loglik) + self.alpha * (self.l1_ratio * l1 + (1.0 - self.l1_ratio) * l2)
        grad = -grad_loglik + self.alpha * (self.l1_ratio * grad_l1 + (1.0 - self.l1_ratio) * beta)
        return float(nll), grad

    def fit(self, X: np.ndarray, time: np.ndarray, event: np.ndarray) -> "CoxElasticNet":
        beta0 = np.zeros(X.shape[1], dtype=float)
        res = minimize(
            fun=lambda b: self._nll_grad(b, X, time, event)[0],
            x0=beta0,
            jac=lambda b: self._nll_grad(b, X, time, event)[1],
            method="L-BFGS-B",
            options={"maxiter": self.maxiter},
        )
        self.coef_ = res.x
        self.success_ = bool(res.success)
        self.n_iter_ = int(res.nit)
        self.message_ = str(res.message)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_


# ---------------------------------------------------------------------------
# Utility metrics
# ---------------------------------------------------------------------------

def concordance_index_censored(times: np.ndarray, events: np.ndarray, risk_scores: np.ndarray) -> float:
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


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = int(len(scores))
    if n == 0:
        return float("inf")
    level = float(np.ceil((n + 1.0) * (1.0 - alpha))) / n
    if level > 1.0:
        return float("inf")
    return float(np.quantile(scores, level))


# ---------------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------------

def prepare_dataframe() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input parquet not found: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH).copy()
    for col in ("OS_MONTHS", "OS_EVENT"):
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

    n_rows = int(len(df))
    n_events = int(df["OS_EVENT"].sum())
    n_censored = n_rows - n_events
    if n_rows < 20:
        raise ValueError(f"Too few rows after cleaning: {n_rows}")
    if n_events == 0 or n_censored == 0:
        raise ValueError("Need both events and censored rows.")

    print(f"Data loaded: {n_rows} rows | {n_events} events | {n_censored} censored | event_rate={n_events / n_rows:.3f}")
    return df


def remove_outliers_iqr(df: pd.DataFrame, columns: Tuple[str, ...] = OUTLIER_COLUMNS, iqr_multiplier: float = OUTLIER_IQR_MULTIPLIER) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    work = df.copy()
    report: Dict[str, Any] = {"initial_rows": int(len(work)), "iqr_multiplier": float(iqr_multiplier), "rules": []}
    for col in columns:
        if col not in work.columns:
            report["rules"].append({"column": col, "skipped": True, "reason": "column_not_found", "removed_rows": 0})
            continue
        s = pd.to_numeric(work[col], errors="coerce")
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = float(q3 - q1)
        if not np.isfinite(iqr) or iqr <= 0:
            report["rules"].append({"column": col, "q1": q1, "q3": q3, "iqr": iqr, "skipped": True, "reason": "non_positive_or_non_finite_iqr", "removed_rows": 0})
            continue
        lo = float(q1 - iqr_multiplier * iqr)
        hi = float(q3 + iqr_multiplier * iqr)
        keep = s.isna() | ((s >= lo) & (s <= hi))
        removed = int((~keep).sum())
        report["rules"].append({"column": col, "q1": q1, "q3": q3, "iqr": iqr, "lower_bound": lo, "upper_bound": hi, "removed_rows": removed, "skipped": False})
        work = work.loc[keep].copy()
        if len(work) == 0:
            raise ValueError(f"Outlier filter removed all rows at column '{col}'.")
    report["final_rows"] = int(len(work))
    report["total_removed"] = int(report["initial_rows"] - report["final_rows"])
    return work, report


# ---------------------------------------------------------------------------
# Splitting and feature selection
# ---------------------------------------------------------------------------

def split_three_way(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Dict[str, Any]]]:
    idx = np.arange(len(df), dtype=int)
    y = df["OS_EVENT"].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError("OS_EVENT has only one class; cannot stratify.")

    idx_train, idx_tmp = train_test_split(idx, test_size=0.4, random_state=RANDOM_STATE, stratify=y)
    idx_cal, idx_test = train_test_split(idx_tmp, test_size=0.5, random_state=RANDOM_STATE, stratify=y[idx_tmp])

    report: Dict[str, Dict[str, Any]] = {}
    for name, sidx in (("train", idx_train), ("calibration", idx_cal), ("test", idx_test)):
        nr = int(len(sidx))
        ne = int(y[sidx].sum())
        report[name] = {"rows": nr, "events": ne, "censored": nr - ne, "event_rate": float(ne / nr) if nr else float("nan")}
        if ne == 0 or (nr - ne) == 0:
            raise ValueError(f"Split '{name}' has only one class.")
    return idx_train, idx_cal, idx_test, report


def get_feature_sets(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    clinical_candidates = ["AGE", "SEX", "RACE", "ETHNICITY", "CANCER_TYPE", "AGE_GROUP"]
    clinical_cols = [c for c in clinical_candidates if c in df.columns]
    expr_cols = sorted(c for c in df.columns if c.startswith("EXPR_"))
    if not clinical_cols and not expr_cols:
        raise ValueError("No usable feature columns found.")
    return clinical_cols, expr_cols


def _empty_drop_table() -> pd.DataFrame:
    return pd.DataFrame(columns=["group", "feature_dropped", "anchor_feature", "abs_correlation", "threshold", "method"])


def _drop_by_full_correlation(numeric_df: pd.DataFrame, threshold: float, group_name: str) -> Tuple[Set[str], pd.DataFrame, Dict[str, Any]]:
    abs_corr = numeric_df.corr(method="pearson").abs()
    if abs_corr.empty:
        return set(), _empty_drop_table(), {"method": "full_correlation_matrix", "n_pairs_above_threshold": 0}
    upper_mask = np.triu(np.ones(abs_corr.shape, dtype=bool), k=1)
    upper = abs_corr.where(upper_mask)
    upper_arr = upper.to_numpy(dtype=float)
    ri, ci = np.where(np.isfinite(upper_arr) & (upper_arr > threshold))
    high_pairs = sorted([(str(upper.index[i]), str(upper.columns[j]), float(upper_arr[i, j])) for i, j in zip(ri, ci)], key=lambda t: t[2], reverse=True)
    drop_rows: List[Dict[str, Any]] = []
    drop_set: Set[str] = set()
    for col in upper.columns:
        high = upper[col][upper[col] > threshold].dropna()
        if high.empty:
            continue
        anchor = str(high.idxmax())
        corr_val = float(high.max())
        drop_set.add(str(col))
        drop_rows.append({"group": group_name, "feature_dropped": str(col), "anchor_feature": anchor, "abs_correlation": corr_val, "threshold": float(threshold), "method": "full_correlation_matrix"})
    preview = [{"feature_a": a, "feature_b": b, "abs_correlation": c} for a, b, c in high_pairs[:20]]
    return drop_set, pd.DataFrame(drop_rows), {"method": "full_correlation_matrix", "n_pairs_above_threshold": int(len(high_pairs)), "top_high_correlation_pairs_preview": preview}


def _drop_by_incremental_correlation(numeric_df: pd.DataFrame, threshold: float, group_name: str) -> Tuple[Set[str], pd.DataFrame, Dict[str, Any]]:
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
            drop_rows.append({"group": group_name, "feature_dropped": str(col_name), "anchor_feature": str(cols[int(k[best_idx])]), "abs_correlation": best_corr, "threshold": float(threshold), "method": "incremental_against_kept"})
        else:
            kept_idx.append(j)
    drop_set = {str(r["feature_dropped"]) for r in drop_rows}
    return drop_set, pd.DataFrame(drop_rows), {"method": "incremental_against_kept", "n_dropped": int(len(drop_rows))}


def drop_collinear_features(df_reference: pd.DataFrame, columns: List[str], threshold: float, group_name: str) -> Tuple[List[str], pd.DataFrame, Dict[str, Any]]:
    if not columns:
        return [], _empty_drop_table(), {"group": group_name, "threshold": float(threshold), "input_feature_count": 0, "numeric_evaluated_count": 0, "dropped_count": 0, "kept_count": 0, "skipped_non_numeric_or_too_sparse": [], "method_details": {"method": "none"}}
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
        return list(columns), _empty_drop_table(), {"group": group_name, "threshold": float(threshold), "input_feature_count": int(len(columns)), "numeric_evaluated_count": int(numeric_df.shape[1]), "dropped_count": 0, "kept_count": int(len(columns)), "skipped_non_numeric_or_too_sparse": skipped_sparse, "skipped_constant_or_near_constant": skipped_constant, "method_details": {"method": "insufficient_numeric_features"}}
    if numeric_df.shape[1] <= COLLINEARITY_MAX_FULL_CORR_FEATURES:
        drop_set, drop_table, method_details = _drop_by_full_correlation(numeric_df, threshold, group_name)
    else:
        drop_set, drop_table, method_details = _drop_by_incremental_correlation(numeric_df, threshold, group_name)
    kept_columns = [c for c in columns if c not in drop_set]
    if drop_table.empty:
        drop_table = _empty_drop_table()
    summary = {"group": group_name, "threshold": float(threshold), "input_feature_count": int(len(columns)), "numeric_evaluated_count": int(numeric_df.shape[1]), "dropped_count": int(len(drop_set)), "kept_count": int(len(kept_columns)), "dropped_features": sorted(drop_set), "skipped_non_numeric_or_too_sparse": skipped_sparse, "skipped_constant_or_near_constant": skipped_constant, "method_details": method_details}
    return kept_columns, drop_table, summary


def apply_collinearity_filter(df_train: pd.DataFrame, clinical_cols: List[str], expr_cols: List[str], threshold: float = COLLINEARITY_THRESHOLD) -> Tuple[List[str], List[str], pd.DataFrame, Dict[str, Any]]:
    expr_kept, expr_drop_table, expr_summary = drop_collinear_features(df_reference=df_train, columns=expr_cols, threshold=threshold, group_name="expression")
    clin_numeric: List[str] = []
    clin_non_numeric: List[str] = []
    for col in clinical_cols:
        (clin_numeric if int(pd.to_numeric(df_train[col], errors="coerce").notna().sum()) >= 2 else clin_non_numeric).append(col)
    clin_num_kept, clin_drop_table, clin_summary = drop_collinear_features(df_reference=df_train, columns=clin_numeric, threshold=threshold, group_name="clinical_numeric")
    clin_dropped = set(clin_summary.get("dropped_features", []))
    clinical_final = [c for c in clinical_cols if c not in clin_dropped]
    all_tables = [t for t in (expr_drop_table, clin_drop_table) if not t.empty]
    dropped_df = pd.concat(all_tables, axis=0, ignore_index=True).sort_values(["group", "abs_correlation", "feature_dropped"], ascending=[True, False, True]) if all_tables else _empty_drop_table()
    summary = {"threshold_abs_pearson": float(threshold), "fit_reference": "train_split_only", "expression": expr_summary, "clinical_numeric": clin_summary, "clinical_non_numeric_or_sparse_kept_as_is": clin_non_numeric, "feature_counts": {"expr_before": int(len(expr_cols)), "expr_after": int(len(expr_kept)), "clinical_before": int(len(clinical_cols)), "clinical_after": int(len(clinical_final)), "total_before": int(len(expr_cols) + len(clinical_cols)), "total_after": int(len(expr_kept) + len(clinical_final)), "total_dropped": int((len(expr_cols) + len(clinical_cols)) - (len(expr_kept) + len(clinical_final)))}}
    return clinical_final, expr_kept, dropped_df, summary


# ---------------------------------------------------------------------------
# Feature preprocessing
# ---------------------------------------------------------------------------

def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _fit_clinical_block(df_tr: pd.DataFrame, df_va: pd.DataFrame, clinical_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, Optional[ColumnTransformer], List[str], List[str]]:
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
    return np.asarray(pre.fit_transform(c_tr), dtype=float), np.asarray(pre.transform(c_va), dtype=float), pre, num_cols, cat_cols


def _fit_expression_block(df_tr: pd.DataFrame, df_va: pd.DataFrame, expr_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, Optional[Pipeline], int]:
    empty = lambda n: np.empty((n, 0), dtype=float)
    if not expr_cols:
        return empty(len(df_tr)), empty(len(df_va)), None, 0
    e_tr = df_tr[expr_cols].copy()
    e_va = df_va[expr_cols].copy()
    n_comp = min(EXPR_PCA_COMPONENTS, int(e_tr.shape[0]), int(e_tr.shape[1]))
    if n_comp < 1:
        raise ValueError("Cannot configure PCA with fewer than 1 component.")
    pipe = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=0.0)), ("sc", StandardScaler()), ("pca", PCA(n_components=n_comp, random_state=RANDOM_STATE))])
    return np.asarray(pipe.fit_transform(e_tr), dtype=float), np.asarray(pipe.transform(e_va), dtype=float), pipe, int(n_comp)


def fit_transform_features(df_tr: pd.DataFrame, df_va: pd.DataFrame, clinical_cols: List[str], expr_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, Optional[ColumnTransformer], Optional[Pipeline], StandardScaler, Dict[str, Any]]:
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
    checks: Dict[str, Any] = {"clinical_num_cols": num_cols, "clinical_cat_cols": cat_cols, "clinical_output_dim": int(Xc_tr.shape[1]), "expr_input_dim": int(len(expr_cols)), "expr_pca_components": int(n_comp), "expr_output_dim": int(Xe_tr.shape[1]), "final_dim": int(X_tr.shape[1]), "final_train_mean_abs_max": float(np.max(np.abs(np.nanmean(X_tr, axis=0)))), "final_train_std_min": float(np.min(stds)), "final_train_std_max": float(np.max(stds)), "final_constant_feature_count": int(np.sum(stds < 1e-12))}
    return X_tr, X_va, clin_pre, expr_pipe, scaler, checks


def transform_features(df: pd.DataFrame, clinical_cols: List[str], expr_cols: List[str], clin_pre: Optional[ColumnTransformer], expr_pipe: Optional[Pipeline], scaler: StandardScaler) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# Cross-validation hyperparameter search
# ---------------------------------------------------------------------------

def run_cv_tuning(df_train: pd.DataFrame, clinical_cols: List[str], expr_cols: List[str]) -> pd.DataFrame:
    y = df_train["OS_EVENT"].to_numpy(dtype=int)
    class_counts = np.bincount(y, minlength=2)
    minority = int(np.min(class_counts[class_counts > 0]))
    if len(class_counts[class_counts > 0]) < 2:
        raise ValueError("CV needs both classes.")
    n_splits = min(CV_FOLDS, minority)
    if n_splits < 2:
        raise ValueError(f"Too few minority-class samples ({minority}) for CV.")

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
                    ci = concordance_index_censored(va["OS_MONTHS"].to_numpy(float), va["OS_EVENT"].to_numpy(int), mdl.predict_risk(X_va))
                    if not np.isfinite(ci):
                        failures.append(f"fold_{fold_id}:non_finite_ci")
                        continue
                    fold_scores.append(float(ci))
                except Exception as exc:
                    failures.append(f"fold_{fold_id}:{type(exc).__name__}")
            rows.append({"alpha": float(cox_alpha), "l1_ratio": float(l1_ratio), "n_valid_folds": int(len(fold_scores)), "n_failed_folds": int(n_splits - len(fold_scores)), "mean_c_index": float(np.mean(fold_scores)) if fold_scores else float("nan"), "std_c_index": float(np.std(fold_scores)) if fold_scores else float("nan"), "fold_scores": ", ".join(f"{s:.4f}" for s in fold_scores), "failure_notes": "; ".join(failures)})

    return pd.DataFrame(rows).sort_values(["n_valid_folds", "mean_c_index", "std_c_index"], ascending=[False, False, True]).reset_index(drop=True)


def select_best_hyperparameters(cv_df: pd.DataFrame) -> pd.Series:
    viable = cv_df[np.isfinite(cv_df["mean_c_index"]) & (cv_df["n_valid_folds"] >= 2)].copy()
    if viable.empty:
        raise ValueError("No viable hyperparameter combination in CV.")
    return viable.sort_values(["n_valid_folds", "mean_c_index", "std_c_index"], ascending=[False, False, True]).iloc[0]


# ---------------------------------------------------------------------------
# Raw-month CQR interval prediction
# ---------------------------------------------------------------------------

def cqr_month_interval(X_tr: np.ndarray, t_tr: np.ndarray, e_tr: np.ndarray, X_cal: np.ndarray, t_cal: np.ndarray, e_cal: np.ndarray, X_te: np.ndarray, t_te: np.ndarray, e_te: np.ndarray, alpha: float = CONFORMAL_ALPHA) -> Tuple[np.ndarray, np.ndarray, float, float, Dict[str, Any]]:
    n_te = len(X_te)
    lo = np.full(n_te, np.nan, dtype=float)
    hi = np.full(n_te, np.nan, dtype=float)

    tr_mask = e_tr == 1
    cal_mask = e_cal == 1
    te_mask = e_te == 1
    n_tr_ev = int(tr_mask.sum())
    n_cal_ev = int(cal_mask.sum())
    n_te_ev = int(te_mask.sum())

    details: Dict[str, Any] = {
        "method": "CQR_Romano_Patterson_Candes_NeurIPS2019",
        "reference": "arXiv:1905.03222",
        "scale": "raw_months_NO_log_transform",
        "alpha": float(alpha),
        "alpha_lo_quantile": float(alpha / 2.0),
        "alpha_hi_quantile": float(1.0 - alpha / 2.0),
        "n_train_events": n_tr_ev,
        "n_cal_events": n_cal_ev,
        "n_test_events": n_te_ev,
        "cqr_min_train_events": CQR_MIN_TRAIN_EVENTS,
        "cqr_min_cal_events": CQR_MIN_CAL_EVENTS,
        "max_interval_width_cap": MAX_INTERVAL_WIDTH_MONTHS,
        "coverage_claimed_for": "event-only_test_rows_under_exchangeability",
        "coverage_not_claimed_for": "censored_test_rows",
        "status": "ok",
    }

    if n_tr_ev < CQR_MIN_TRAIN_EVENTS:
        details["status"] = f"skipped: only {n_tr_ev} training events (need >= {CQR_MIN_TRAIN_EVENTS})"
        return lo, hi, np.nan, np.nan, details
    if n_cal_ev < CQR_MIN_CAL_EVENTS:
        details["status"] = f"skipped: only {n_cal_ev} calibration events (need >= {CQR_MIN_CAL_EVENTS})"
        return lo, hi, np.nan, np.nan, details

    X_tr_ev = X_tr[tr_mask]
    y_tr_ev = t_tr[tr_mask]
    X_cal_ev = X_cal[cal_mask]
    y_cal_ev = t_cal[cal_mask]

    alpha_lo = alpha / 2.0
    alpha_hi = 1.0 - alpha / 2.0
    try:
        qr_lo = GradientBoostingRegressor(alpha=alpha_lo, **CQR_GBR_PARAMS)
        qr_hi = GradientBoostingRegressor(alpha=alpha_hi, **CQR_GBR_PARAMS)
        qr_lo.fit(X_tr_ev, y_tr_ev)
        qr_hi.fit(X_tr_ev, y_tr_ev)
    except Exception as exc:
        details["status"] = f"skipped: QR fit failed ({type(exc).__name__}: {exc})"
        return lo, hi, np.nan, np.nan, details

    q_lo_cal_raw = qr_lo.predict(X_cal_ev)
    q_hi_cal_raw = qr_hi.predict(X_cal_ev)
    q_lo_cal = np.minimum(q_lo_cal_raw, q_hi_cal_raw)
    q_hi_cal = np.maximum(q_lo_cal_raw, q_hi_cal_raw)
    scores = np.maximum(q_lo_cal - y_cal_ev, y_cal_ev - q_hi_cal)

    details["calibration_score_min"] = float(np.min(scores))
    details["calibration_score_p25"] = float(np.percentile(scores, 25))
    details["calibration_score_median"] = float(np.median(scores))
    details["calibration_score_p75"] = float(np.percentile(scores, 75))
    details["calibration_score_p90"] = float(np.quantile(scores, 0.90))
    details["calibration_score_max"] = float(np.max(scores))
    details["calibration_score_mean"] = float(np.mean(scores))
    details["raw_qr_interval_width_mean_cal_months"] = float(np.mean(q_hi_cal - q_lo_cal))

    qhat = conformal_quantile(scores, alpha)
    details["qhat_months"] = float(qhat)
    if not np.isfinite(qhat):
        details["status"] = f"skipped: qhat=+inf, calibration set too small (n_cal_events={n_cal_ev}, alpha={alpha})"
        return lo, hi, np.nan, np.nan, details

    q_lo_te_raw = qr_lo.predict(X_te)
    q_hi_te_raw = qr_hi.predict(X_te)
    q_lo_te = np.minimum(q_lo_te_raw, q_hi_te_raw)
    q_hi_te = np.maximum(q_lo_te_raw, q_hi_te_raw)

    lo = np.clip(q_lo_te - qhat, 0.0, None)
    hi = np.maximum(lo, q_hi_te + qhat)

    if MAX_INTERVAL_WIDTH_MONTHS is not None:
        cap = float(MAX_INTERVAL_WIDTH_MONTHS)
        hi = np.minimum(hi, lo + cap)
        details["max_interval_width_cap_applied"] = True
    else:
        details["max_interval_width_cap_applied"] = False

    widths = hi - lo
    details["mean_interval_width_months_all_test"] = float(np.mean(widths))
    details["median_interval_width_months_all_test"] = float(np.median(widths))
    details["p90_interval_width_months_all_test"] = float(np.percentile(widths, 90))

    if n_te_ev > 0:
        lo_ev = lo[te_mask]
        hi_ev = hi[te_mask]
        y_te_ev = t_te[te_mask]
        coverage = float(np.mean((y_te_ev >= lo_ev) & (y_te_ev <= hi_ev)))
        ev_widths = hi_ev - lo_ev
        details["coverage_test_events_only"] = coverage
        details["mean_interval_width_months_test_events"] = float(np.mean(ev_widths))
        details["median_interval_width_months_test_events"] = float(np.median(ev_widths))
        details["p90_interval_width_months_test_events"] = float(np.percentile(ev_widths, 90))
    else:
        coverage = float("nan")
        details["coverage_test_events_only"] = float("nan")

    return lo, hi, float(qhat), coverage, details


# ---------------------------------------------------------------------------
# Time-dependent conformal horizon classification
# ---------------------------------------------------------------------------

def _horizon_labels(time_arr: np.ndarray, event_arr: np.ndarray, horizon: float) -> Tuple[np.ndarray, np.ndarray]:
    confirmed_event = (event_arr == 1) & (time_arr <= horizon)
    confirmed_no_evt = time_arr > horizon
    keep = confirmed_event | confirmed_no_evt
    y = np.where(confirmed_event[keep], 1, 0)
    return keep, y


def time_dependent_conformal(r_cal: np.ndarray, t_cal: np.ndarray, e_cal: np.ndarray, r_te: np.ndarray, t_te: np.ndarray, e_te: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, Dict[str, Any]]]:
    rows_metrics: List[Dict[str, Any]] = []
    rows_pred: List[Dict[str, Any]] = []
    rows_cls: List[Dict[str, Any]] = []
    calibrators: Dict[int, Dict[str, Any]] = {}

    for h in HORIZONS_MONTHS:
        keep_cal, y_cal = _horizon_labels(t_cal, e_cal, h)
        keep_te, y_te = _horizon_labels(t_te, e_te, h)
        n_cal = int(len(y_cal))
        n_te = int(len(y_te))
        base: Dict[str, Any] = {"horizon_months": h, "n_cal_known": n_cal, "n_test_known": n_te}

        if n_cal == 0 or n_te == 0:
            rows_metrics.append({**base, "note": "no_known_labels"})
            continue
        if len(np.unique(y_cal)) < 2:
            rows_metrics.append({**base, "note": "insufficient_cal_class_variation"})
            continue
        if len(np.unique(y_te)) < 2:
            rows_metrics.append({**base, "note": "insufficient_test_class_variation"})
            continue

        try:
            iso = IsotonicRegression(out_of_bounds="clip")
            p_cal = iso.fit_transform(r_cal[keep_cal], y_cal)
            p_te = iso.predict(r_te[keep_te])
        except Exception as exc:
            rows_metrics.append({**base, "note": f"isotonic_failed:{type(exc).__name__}"})
            continue

        if not (np.isfinite(p_cal).all() and np.isfinite(p_te).all()):
            rows_metrics.append({**base, "note": "non_finite_isotonic_probs"})
            continue

        scores = np.abs(y_cal - p_cal)
        qhat_h = conformal_quantile(scores, CONFORMAL_ALPHA)
        if not np.isfinite(qhat_h):
            rows_metrics.append({**base, "note": "non_finite_conformal_quantile"})
            continue

        p_lo = np.clip(p_te - qhat_h, 0.0, 1.0)
        p_hi = np.clip(p_te + qhat_h, 0.0, 1.0)
        coverage = float(np.mean((y_te >= p_lo) & (y_te <= p_hi)))
        rows_metrics.append({**base, "qhat": float(qhat_h), "coverage_known_test": coverage, "mean_interval_width": float(np.mean(p_hi - p_lo)), "note": "ok"})

        best_thr = 0.5
        best_j = -1.0
        for thr in np.linspace(0, 1, 201):
            yhat_c = (p_cal >= thr).astype(int)
            tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_cal, yhat_c, labels=[0, 1]).ravel()
            tpr = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
            tnr = tn_c / (tn_c + fp_c) if (tn_c + fp_c) > 0 else 0.0
            j = tpr + tnr - 1.0
            if j > best_j:
                best_j = j
                best_thr = float(thr)

        yhat_t = (p_te >= best_thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, yhat_t, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        try:
            roc_auc = float(roc_auc_score(y_te, p_te))
        except Exception:
            roc_auc = float("nan")
        try:
            pr_auc = float(average_precision_score(y_te, p_te))
        except Exception:
            pr_auc = float("nan")

        rows_cls.append({"horizon_months": h, "n_test_known": n_te, "event_rate_test_known": float(np.mean(y_te)), "threshold_from_calibration": best_thr, "roc_auc": roc_auc, "pr_auc": pr_auc, "brier": float(brier_score_loss(y_te, p_te)), "accuracy": float(accuracy_score(y_te, yhat_t)), "precision": float(precision_score(y_te, yhat_t, zero_division=0)), "recall_sensitivity": float(recall_score(y_te, yhat_t, zero_division=0)), "specificity": float(specificity), "f1": float(f1_score(y_te, yhat_t, zero_division=0)), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)})
        calibrators[int(h)] = {"isotonic": iso, "qhat": float(qhat_h), "threshold": float(best_thr)}
        for i_local, i_global in enumerate(np.where(keep_te)[0]):
            rows_pred.append({"horizon_months": h, "test_row_index_within_split": int(i_global), "risk_score": float(r_te[i_global]), "y_true_by_horizon": int(y_te[i_local]), "p_event_hat": float(p_te[i_local]), "p_event_lo": float(p_lo[i_local]), "p_event_hi": float(p_hi[i_local])})

    return pd.DataFrame(rows_metrics), pd.DataFrame(rows_pred), pd.DataFrame(rows_cls), calibrators


# ---------------------------------------------------------------------------
# Monte Carlo survival summaries (new layer)
# ---------------------------------------------------------------------------

def fit_breslow_baseline_hazard(times: np.ndarray, events: np.ndarray, risk_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
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
        if first_idx >= len(times):
            continue
        at_risk = float(risk_suffix[first_idx])
        d = int(np.sum((times == t) & (events == 1)))
        if at_risk <= 0 or d <= 0:
            continue
        cumhaz += float(d / at_risk)
        base_times.append(float(t))
        base_cumhaz.append(float(cumhaz))

    meta = {"status": "ok" if base_times else "skipped:no_event_times", "n_rows": int(len(times)), "n_events": int(events.sum()), "n_unique_event_times": int(len(base_times)), "max_observed_followup_months": float(np.max(times)), "max_event_time_months": float(base_times[-1]) if base_times else float("nan")}
    return np.asarray(base_times, dtype=float), np.asarray(base_cumhaz, dtype=float), meta


def simulate_cox_survival_times(risk_scores: np.ndarray, baseline_times: np.ndarray, baseline_cumhaz: np.ndarray, max_followup_months: float, n_sims: int = MC_N_SIMS, random_state: int = MC_RANDOM_STATE) -> np.ndarray:
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


def monte_carlo_survival_block(t_tr: np.ndarray, e_tr: np.ndarray, r_tr: np.ndarray, t_te: np.ndarray, e_te: np.ndarray, r_te: np.ndarray, patient_ids_te: Optional[np.ndarray] = None, n_sims: int = MC_N_SIMS) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    baseline_times, baseline_cumhaz, baseline_meta = fit_breslow_baseline_hazard(t_tr, e_tr, r_tr)
    sim_times = simulate_cox_survival_times(r_te, baseline_times=baseline_times, baseline_cumhaz=baseline_cumhaz, max_followup_months=baseline_meta.get("max_observed_followup_months", float(np.nan)), n_sims=n_sims, random_state=MC_RANDOM_STATE)

    if sim_times.size == 0:
        empty = pd.DataFrame(columns=["test_row_index_within_split", "PATIENT_ID"])
        return empty, baseline_meta, {"status": "skipped:empty_test_set"}

    rows = []
    for i in range(sim_times.shape[0]):
        s = sim_times[i]
        rows.append({
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
        })

    baseline_meta = {**baseline_meta, "baseline_times": baseline_times, "baseline_cumhaz": baseline_cumhaz}
    summary = {
        "status": "ok",
        "n_test": int(sim_times.shape[0]),
        "n_sims": int(n_sims),
        "probability_horizons_months": list(MC_HORIZONS_MONTHS),
        "rmst_horizon_months": float(MC_RMST_HORIZON_MONTHS),
        "interpretation": "bounded Monte Carlo summaries from the fitted Cox model; p90 is an upper-plausible survival time, not a guarantee",
    }
    return pd.DataFrame(rows), baseline_meta, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_dataframe().reset_index(drop=True)
    df_no, outlier_report = remove_outliers_iqr(df)
    print(f"After outlier filter: {len(df_no)} rows (removed {len(df) - len(df_no)})")

    idx_train, idx_cal, idx_test, split_report = split_three_way(df_no)
    clinical_cols, expr_cols = get_feature_sets(df_no)
    print(f"Clinical features: {len(clinical_cols)} | Expression features: {len(expr_cols)}")

    df_train_for_filter = df_no.iloc[idx_train].copy().reset_index(drop=True)
    clinical_cols_f, expr_cols_f, dropped_df, coll_summary = apply_collinearity_filter(df_train_for_filter, clinical_cols, expr_cols, threshold=COLLINEARITY_THRESHOLD)
    dropped_df.to_csv(OUT_DIR / "collinearity_dropped_features.csv", index=False)
    (OUT_DIR / "collinearity_summary.json").write_text(json.dumps(coll_summary, indent=2), encoding="utf-8")
    print(f"After collinearity: clinical {len(clinical_cols)} → {len(clinical_cols_f)} | expr {len(expr_cols)} → {len(expr_cols_f)}")
    if not clinical_cols_f and not expr_cols_f:
        raise ValueError("All features removed by collinearity filter.")

    selected = set(clinical_cols_f) | set(expr_cols_f)
    forbidden_exact = {"OS_MONTHS", "OS_EVENT", "OS_STATUS", "DFS_STATUS", "DFS_EVENT"}
    forbidden_prefixes = ("OS_", "DFS_")
    leaking = sorted(c for c in selected if c in forbidden_exact or any(c.startswith(p) for p in forbidden_prefixes))
    if leaking:
        raise ValueError(f"Target-leakage features detected: {leaking}")

    df_train = df_no.iloc[idx_train].copy().reset_index(drop=True)
    print("Running CV hyperparameter search …")
    cv_df = run_cv_tuning(df_train, clinical_cols_f, expr_cols_f)
    cv_df.to_csv(OUT_DIR / "hyperparameter_cv_results.csv", index=False)
    best = select_best_hyperparameters(cv_df)
    best_cox_alpha = float(best["alpha"])
    best_l1 = float(best["l1_ratio"])
    print(f"Best params: alpha={best_cox_alpha}, l1_ratio={best_l1}, CV c-index={float(best['mean_c_index']):.4f}")

    tr = df_no.iloc[idx_train].copy()
    cal = df_no.iloc[idx_cal].copy()
    te = df_no.iloc[idx_test].copy()

    X_tr, X_cal, clin_pre, expr_pipe, scaler, feat_checks = fit_transform_features(tr, cal, clinical_cols_f, expr_cols_f)
    X_te = transform_features(te, clinical_cols_f, expr_cols_f, clin_pre, expr_pipe, scaler)

    t_tr = tr["OS_MONTHS"].to_numpy(float)
    e_tr = tr["OS_EVENT"].to_numpy(int)
    t_cal = cal["OS_MONTHS"].to_numpy(float)
    e_cal = cal["OS_EVENT"].to_numpy(int)
    t_te = te["OS_MONTHS"].to_numpy(float)
    e_te = te["OS_EVENT"].to_numpy(int)

    print("Fitting final Cox model …")
    model = CoxElasticNet(alpha=best_cox_alpha, l1_ratio=best_l1, maxiter=MAXITER)
    model.fit(X_tr, t_tr, e_tr)
    if not bool(getattr(model, "success_", True)):
        raise RuntimeError(f"Cox did not converge: {getattr(model, 'message_', 'unknown')}")

    r_tr = model.predict_risk(X_tr)
    r_cal = model.predict_risk(X_cal)
    r_te = model.predict_risk(X_te)

    ci_train = float(concordance_index_censored(t_tr, e_tr, r_tr))
    ci_cal = float(concordance_index_censored(t_cal, e_cal, r_cal))
    ci_test = float(concordance_index_censored(t_te, e_te, r_te))
    print(f"C-index: train={ci_train:.4f} | cal={ci_cal:.4f} | test={ci_test:.4f}")

    print("Computing CQR month intervals (raw-month scale, no log transform) …")
    lo, hi, qhat, coverage_ev, cqr_details = cqr_month_interval(X_tr, t_tr, e_tr, X_cal, t_cal, e_cal, X_te, t_te, e_te, alpha=CONFORMAL_ALPHA)
    print(f"CQR status: {cqr_details['status']}")
    if np.isfinite(qhat):
        print(f"  qhat = {qhat:.2f} months")
    if np.isfinite(cqr_details.get("mean_interval_width_months_all_test", np.nan)):
        print(f"  Mean interval width (all test): {cqr_details['mean_interval_width_months_all_test']:.1f} months")
    if np.isfinite(coverage_ev):
        print(f"  Empirical coverage (event rows): {coverage_ev:.3f}")

    print("Running Monte Carlo survival summaries …")
    mc_te, mc_baseline_meta, mc_summary = monte_carlo_survival_block(t_tr, e_tr, r_tr, t_te, e_te, r_te, patient_ids_te=te["PATIENT_ID"].astype(str).to_numpy(), n_sims=MC_N_SIMS)
    mc_te.to_csv(OUT_DIR / "monte_carlo_survival_test_predictions.csv", index=False)

    td_metrics, td_preds, td_cls, td_calibrators = time_dependent_conformal(r_cal, t_cal, e_cal, r_te, t_te, e_te)
    td_metrics.to_csv(OUT_DIR / "time_dependent_conformal_metrics.csv", index=False)
    td_preds.to_csv(OUT_DIR / "time_dependent_conformal_test_predictions.csv", index=False)
    td_cls.to_csv(OUT_DIR / "time_dependent_horizon_classification_metrics.csv", index=False)

    patient_ids = pd.concat([tr["PATIENT_ID"], cal["PATIENT_ID"], te["PATIENT_ID"]], axis=0).astype(str).values
    pred_df = pd.DataFrame({
        "split": (["train"] * len(tr) + ["calibration"] * len(cal) + ["test"] * len(te)),
        "PATIENT_ID": patient_ids,
        "OS_MONTHS": np.concatenate([t_tr, t_cal, t_te]),
        "OS_EVENT": np.concatenate([e_tr, e_cal, e_te]),
        "risk_score": np.concatenate([r_tr, r_cal, r_te]),
        "pred_survival_lo_months_90": np.concatenate([np.full(len(tr), np.nan), np.full(len(cal), np.nan), lo]),
        "pred_survival_hi_months_90": np.concatenate([np.full(len(tr), np.nan), np.full(len(cal), np.nan), hi]),
        "mc_survival_p10_months": np.concatenate([np.full(len(tr) + len(cal), np.nan), mc_te["mc_survival_p10_months"].to_numpy(dtype=float) if not mc_te.empty else np.array([], dtype=float)]),
        "mc_survival_p50_months": np.concatenate([np.full(len(tr) + len(cal), np.nan), mc_te["mc_survival_p50_months"].to_numpy(dtype=float) if not mc_te.empty else np.array([], dtype=float)]),
        "mc_survival_p90_months": np.concatenate([np.full(len(tr) + len(cal), np.nan), mc_te["mc_survival_p90_months"].to_numpy(dtype=float) if not mc_te.empty else np.array([], dtype=float)]),
        "mc_prob_survive_12_months": np.concatenate([np.full(len(tr) + len(cal), np.nan), mc_te["mc_prob_survive_12_months"].to_numpy(dtype=float) if not mc_te.empty else np.array([], dtype=float)]),
        "mc_prob_survive_24_months": np.concatenate([np.full(len(tr) + len(cal), np.nan), mc_te["mc_prob_survive_24_months"].to_numpy(dtype=float) if not mc_te.empty else np.array([], dtype=float)]),
        "mc_prob_survive_36_months": np.concatenate([np.full(len(tr) + len(cal), np.nan), mc_te["mc_prob_survive_36_months"].to_numpy(dtype=float) if not mc_te.empty else np.array([], dtype=float)]),
        "mc_prob_survive_60_months": np.concatenate([np.full(len(tr) + len(cal), np.nan), mc_te["mc_prob_survive_60_months"].to_numpy(dtype=float) if not mc_te.empty else np.array([], dtype=float)]),
        "mc_rmst_60_months": np.concatenate([np.full(len(tr) + len(cal), np.nan), mc_te["mc_rmst_60_months"].to_numpy(dtype=float) if not mc_te.empty else np.array([], dtype=float)]),
    })
    pred_df.to_csv(OUT_DIR / "main_predictions.csv", index=False)
    pred_df.to_csv(OUT_DIR / "tuned_model_predictions.csv", index=False)

    coef_df = pd.DataFrame({"coef_index": np.arange(len(model.coef_)), "coef_value": model.coef_}).sort_values("coef_value", key=np.abs, ascending=False)
    coef_df.to_csv(OUT_DIR / "coefficient_exports.csv", index=False)
    coef_df.to_csv(OUT_DIR / "tuned_model_coefficients.csv", index=False)

    finite_widths = (hi - lo)[np.isfinite(hi - lo)]
    mean_w_all = float(np.mean(finite_widths)) if len(finite_widths) > 0 else float("nan")
    median_w_all = float(np.median(finite_widths)) if len(finite_widths) > 0 else float("nan")
    p90_w_all = float(np.percentile(finite_widths, 90)) if len(finite_widths) > 0 else float("nan")

    metrics: Dict[str, Any] = {
        "input_file": str(INPUT_PATH),
        "best_cox_params": {"alpha": best_cox_alpha, "l1_ratio": best_l1},
        "cv_best_mean_c_index": float(best["mean_c_index"]),
        "cv_best_std_c_index": float(best["std_c_index"]),
        "cv_best_valid_folds": int(best["n_valid_folds"]),
        "n_total_after_outlier_filter": int(len(df_no)),
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
        "collinearity_filter": {"threshold_abs_pearson": COLLINEARITY_THRESHOLD, "clinical_before": int(len(clinical_cols)), "clinical_after": int(len(clinical_cols_f)), "expr_before": int(len(expr_cols)), "expr_after": int(len(expr_cols_f))},
        "cqr_month_interval": {**cqr_details, "mean_interval_width_months_all_test": mean_w_all, "median_interval_width_months_all_test": median_w_all, "p90_interval_width_months_all_test": p90_w_all},
        "monte_carlo": mc_summary,
        "time_dependent_conformal": {"horizons_months": HORIZONS_MONTHS, "n_horizons_reported": int(td_metrics.shape[0]), "n_horizons_with_models": int(len(td_calibrators))},
    }
    (OUT_DIR / "tuned_model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    consistency: Dict[str, Any] = {
        "input_file": str(INPUT_PATH),
        "rows_before_outlier_filter": int(len(df)),
        "rows_after_outlier_filter": int(len(df_no)),
        "outlier_report": outlier_report,
        "split_report": split_report,
        "feature_checks": feat_checks,
        "feature_counts": {"clinical_before_collinearity": int(len(clinical_cols)), "expr_before_collinearity": int(len(expr_cols)), "clinical_after_collinearity": int(len(clinical_cols_f)), "expr_after_collinearity": int(len(expr_cols_f))},
        "collinearity_summary": coll_summary,
        "leakage_check_passed": True,
        "monte_carlo_baseline_meta": {k: v for k, v in mc_baseline_meta.items() if k not in {"baseline_times", "baseline_cumhaz"}},
    }
    (OUT_DIR / "consistency_checks.json").write_text(json.dumps(consistency, indent=2), encoding="utf-8")
    (OUT_DIR / "audit.json").write_text(json.dumps(consistency, indent=2), encoding="utf-8")

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
        "cqr_qhat": qhat,
        "cqr_alpha": CONFORMAL_ALPHA,
        "cqr_details": cqr_details,
        "time_dependent_calibrators": td_calibrators,
        "horizons_months": HORIZONS_MONTHS,
        "mc_summary": mc_summary,
        "mc_baseline_meta": {k: v for k, v in mc_baseline_meta.items() if k not in {"baseline_times", "baseline_cumhaz"}},
    }
    with open(OUT_DIR / "final_locked_model.pkl", "wb") as f:
        pickle.dump(artifact, f)

    ok = cqr_details["status"] == "ok"
    readme = [
        "# Cox ENet + raw-month CQR + Monte Carlo",
        "",
        "## What this pipeline reports",
        "- risk score from the fitted Cox Elastic-Net model",
        "- conformal interval on raw months",
        "- Monte Carlo survival summaries at fixed horizons",
        "- bounded RMST at 60 months",
        "",
        "## Important interpretation note",
        "The Monte Carlo block does not predict an exact survival lifespan.",
        "It gives simulation-based summaries under the fitted Cox model and",
        "the estimated Breslow baseline hazard.",
        "",
        "The p90 simulated survival month is an upper-plausible time, not a",
        "guarantee that the patient will survive that long.",
        "",
        "## Results",
        f"Test c-index      : {ci_test:.4f}",
        f"CV mean c-index   : {float(best['mean_c_index']):.4f}",
        f"CQR status        : {cqr_details['status']}",
        (f"qhat              : {qhat:.2f} months" if ok and np.isfinite(qhat) else "qhat              : N/A"),
        (f"Coverage (events) : {coverage_ev:.4f}" if ok and np.isfinite(coverage_ev) else "Coverage          : N/A"),
        (f"Mean width (all)  : {mean_w_all:.1f} months" if np.isfinite(mean_w_all) else "Mean width        : N/A"),
        "",
        "## Monte Carlo outputs",
        "- mc_prob_survive_12_months",
        "- mc_prob_survive_24_months",
        "- mc_prob_survive_36_months",
        "- mc_prob_survive_60_months",
        "- mc_survival_p10_months / mc_survival_p50_months / mc_survival_p90_months",
        "- mc_rmst_60_months",
        "",
        "## Files",
        "- main_predictions.csv",
        "- monte_carlo_survival_test_predictions.csv",
        "- tuned_model_predictions.csv",
        "- tuned_model_coefficients.csv",
        "- tuned_model_metrics.json",
        "- metrics.json",
        "- consistency_checks.json",
        "- audit.json",
        "- final_locked_model.pkl",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print("\nDone.")
    print(f"Results → {OUT_DIR}")


if __name__ == "__main__":
    main()
