"""
Cox Elastic-Net + Conformalized Quantile Regression (CQR) — complete pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESIGN RATIONALE AND HONEST LIMITATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Goal
----
Predict how many months a patient is *likely to survive* and attach an upper
bound that says "with 90% confidence the patient will survive AT LEAST this
many months" — the hi bound — or stated differently, produce an interval
[lo, hi] such that the true event time lies in [lo, hi] with ~90% probability
for patients whose event we can observe.

Why CQR (not the old IsotonicRegression + log/expm1 approach)
-------------------------------------------------------------
The previous code computed conformal residuals in log1p-space then applied
expm1 to back-transform. Because expm1 is convex, a single qhat value of ~1.5
in log-space maps asymmetrically: lower bound shrinks moderately, upper bound
explodes (e.g. expm1(3.5 + 1.5) ≈ 402 months). This is the root cause of the
reported wide intervals.

CQR (Romano, Patterson & Candès, NeurIPS 2019, arXiv:1905.03222) avoids this
entirely by operating in the *raw month scale*:
  - Two GradientBoostingRegressor quantile models estimate lo/hi directly.
  - Nonconformity score: E_i = max(q_lo(X_i) − Y_i,  Y_i − q_hi(X_i))
  - Conformal correction qhat is in months, applied symmetrically:
      final_lo = q_lo(X) − qhat,   final_hi = q_hi(X) + qhat
  Both bounds shift by the same absolute number of months → no asymmetric
  explosion.

Critical limitation: censoring breaks exchangeability
------------------------------------------------------
Standard CQR assumes Y is fully observed for calibration points. In survival
data, many patients are censored: we observe min(T, C) not T. Using censored
OS_MONTHS values as regression targets would mix true event times with
administrative cutoff times, contaminating the quantile models and the
nonconformity scores.

Our pragmatic solution: fit and calibrate exclusively on event-only rows
(OS_EVENT == 1). These patients have a directly observed T = OS_MONTHS.
Exchangeability then holds within the event sub-population under i.i.d.
sampling, giving a valid marginal coverage guarantee for future patients who
*do* experience the event. Coverage for censored patients is not claimed.

This is a pragmatic, not a theoretically optimal solution. The literature
(Candès et al. 2023 JRSSS-B, Qin et al. 2024 arXiv:2408.06539) shows that
fully rigorous handling of censoring requires either Type-I censoring with
covariate-shift weighting or bootstrap-resampling approaches. We do not
implement those here because they require censoring-time observations that
may not be available in the transformed dataset.

If your event rate is low (< ~30%), the event-only calibration set will be
small and qhat may be large or +inf. The code reports this explicitly.

What the output means
---------------------
pred_months_lo_90 : estimated lower bound — patient expected to survive at
                    least this many months (model-driven lower quantile minus
                    conformal margin). Clipped to [0, ∞).
pred_months_hi_90 : estimated upper bound — patient unlikely to survive
                    beyond this many months (model-driven upper quantile plus
                    conformal margin). Clipped to [0, ∞).

Claimed coverage: approximately 90% for patients who experience the event,
over repeated sampling from the same distribution. NOT guaranteed for
censored patients.

Wide-interval root cause note
------------------------------
If the intervals are still wide, the most likely explanations (in order) are:
  1. Low event rate → few calibration events → large qhat to achieve 90%.
  2. High heterogeneity in event times → the GBR quantile models have wide
     raw intervals before conformal adjustment.
  3. Mathematical transformation of data: if OS_MONTHS was derived from a
     longitudinal model rather than directly observed, the variance of the
     transformed variable may be high. There is no algorithmic fix for this —
     it reflects genuine uncertainty in the data.

References
----------
Romano Y, Patterson E, Candès EJ (2019). Conformalized Quantile Regression.
  NeurIPS 32. arXiv:1905.03222.
Lei J, G'Sell M, Rinaldo A, Tibshirani RJ, Wasserman L (2018).
  Distribution-free predictive inference for regression. JASA 113:1094-1111.
Candès E, Lei L, Ren Z (2023). Conformalized survival analysis.
  JRSSS-B 85(1):24-45.
Sesia M, Candès EJ (2020). A comparison of some conformal quantile
  regression methods. Stat 9(1):e261.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
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

from train_cox_enet_conformal import CoxElasticNet, concordance_index_censored

# ──────────────────────────────────────────────────────────────────────────────
# Global constants
# ──────────────────────────────────────────────────────────────────────────────

RANDOM_STATE = 42
EXPR_PCA_COMPONENTS = 50
MAXITER = 400
CV_FOLDS = 5

# Miscoverage α: we target (1 − α) = 90% marginal coverage.
CONFORMAL_ALPHA = 0.10

HORIZONS_MONTHS = [12, 24, 36, 60]

# Cox regularisation grid
ALPHA_GRID = [0.1, 0.3, 0.8, 1.5, 3.0]
L1_GRID    = [0.0, 0.1, 0.3, 0.5, 0.7]

OUTLIER_IQR_MULTIPLIER = 3.0
OUTLIER_COLUMNS = ("OS_MONTHS", "AGE")

COLLINEARITY_THRESHOLD = 0.75
COLLINEARITY_MAX_FULL_CORR_FEATURES = 3000

# Quantile regression model for CQR.
# GradientBoostingRegressor with loss='quantile' directly minimises the
# pinball / check loss at level `alpha` (sklearn docs, confirmed).
# Shared params; each model differs only in `alpha`.
CQR_GBR_PARAMS: Dict[str, Any] = {
    "loss": "quantile",
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_samples_leaf": 9,
    "min_samples_split": 9,
    "random_state": RANDOM_STATE,
}

# Minimum event-only rows required in training / calibration for CQR.
# Below these counts the quantile models or the conformal correction are
# unreliable (too few points to estimate a tail quantile accurately).
CQR_MIN_TRAIN_EVENTS = 20
CQR_MIN_CAL_EVENTS   = 10

BASE       = Path("/home/illionar/Projects/ml_research")
INPUT_PATH = BASE / "data" / "preprocessed_cleaned" / "patient_multiomic_cleaned.parquet"
OUT_DIR    = (
    BASE / "data" / "model_outputs"
    / "cox_enet_cqr_v2"
)


# ──────────────────────────────────────────────────────────────────────────────
# Utility: one-hot encoder compatibility shim
# ──────────────────────────────────────────────────────────────────────────────

def make_one_hot_encoder() -> OneHotEncoder:
    """Dense OHE that works across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# ──────────────────────────────────────────────────────────────────────────────
# Utility: conservative conformal quantile  (Lei et al. 2018)
# ──────────────────────────────────────────────────────────────────────────────

def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Conservative finite-sample (1-alpha) quantile of nonconformity scores.

    Formula: level = ceil((n+1)*(1-alpha)) / n
    This is the standard split-conformal correction (Lei et al. 2018,
    Romano et al. 2019). When level > 1 the calibration set is too small
    to achieve the requested coverage and we return +inf.

    Parameters
    ----------
    scores : 1-D array of nonconformity scores (any real values)
    alpha  : miscoverage rate in (0, 1)

    Returns
    -------
    float qhat, possibly +inf
    """
    n = int(len(scores))
    if n == 0:
        return np.inf
    # ceil formula — do the arithmetic in float to avoid int overflow
    level = float(np.ceil((n + 1.0) * (1.0 - alpha))) / n
    if level > 1.0:
        # Not enough calibration points for requested alpha
        return np.inf
    return float(np.quantile(scores, level))


# ──────────────────────────────────────────────────────────────────────────────
# Data loading and basic cleaning
# ──────────────────────────────────────────────────────────────────────────────

def prepare_dataframe() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input parquet not found: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)
    required = {"OS_MONTHS", "OS_EVENT"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["OS_MONTHS"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    df["OS_EVENT"]  = pd.to_numeric(df["OS_EVENT"],  errors="coerce")
    df = df.dropna(subset=["OS_MONTHS", "OS_EVENT"]).copy()
    df = df[df["OS_MONTHS"] > 0].copy()
    df["OS_EVENT"] = (df["OS_EVENT"] > 0).astype(int)

    if "PATIENT_ID" not in df.columns:
        df["PATIENT_ID"] = [f"ROW_{i:07d}" for i in range(len(df))]
    df["PATIENT_ID"] = df["PATIENT_ID"].astype(str)

    n_rows    = int(len(df))
    n_events  = int(df["OS_EVENT"].sum())
    n_censored = n_rows - n_events
    if n_rows < 20:
        raise ValueError(f"Too few rows after cleaning: {n_rows}.")
    if n_events == 0 or n_censored == 0:
        raise ValueError("Only one class present; need both events and censored rows.")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Outlier removal
# ──────────────────────────────────────────────────────────────────────────────

def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: Tuple[str, ...] = OUTLIER_COLUMNS,
    iqr_multiplier: float = OUTLIER_IQR_MULTIPLIER,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    IQR fence: keep rows where col is in [Q1 - k*IQR, Q3 + k*IQR].
    Columns that are non-numeric or have IQR==0 are skipped (reported).
    """
    work = df.copy()
    report: Dict[str, Any] = {
        "initial_rows": int(len(work)),
        "iqr_multiplier": float(iqr_multiplier),
        "rules": [],
    }

    for col in columns:
        if col not in work.columns:
            report["rules"].append({"column": col, "skipped": True,
                                    "reason": "column_not_found", "removed_rows": 0})
            continue

        s = pd.to_numeric(work[col], errors="coerce")
        q1  = float(s.quantile(0.25))
        q3  = float(s.quantile(0.75))
        iqr = float(q3 - q1)

        if not np.isfinite(iqr) or iqr <= 0:
            report["rules"].append({"column": col, "q1": q1, "q3": q3, "iqr": iqr,
                                    "skipped": True,
                                    "reason": "non_positive_or_non_finite_iqr",
                                    "removed_rows": 0})
            continue

        lo   = float(q1 - iqr_multiplier * iqr)
        hi   = float(q3 + iqr_multiplier * iqr)
        keep = s.isna() | ((s >= lo) & (s <= hi))
        removed = int((~keep).sum())
        report["rules"].append({"column": col, "q1": q1, "q3": q3, "iqr": iqr,
                                 "lower_bound": lo, "upper_bound": hi,
                                 "removed_rows": removed, "skipped": False})
        work = work.loc[keep].copy()
        if len(work) == 0:
            raise ValueError(f"Outlier filter removed all rows at column '{col}'.")

    report["final_rows"]    = int(len(work))
    report["total_removed"] = int(report["initial_rows"] - report["final_rows"])
    return work, report


# ──────────────────────────────────────────────────────────────────────────────
# Three-way stratified split
# ──────────────────────────────────────────────────────────────────────────────

def split_three_way(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Dict[str, Any]]]:
    """
    60% train / 20% calibration / 20% test, stratified by OS_EVENT.

    The calibration split serves two purposes:
      (a) conformal calibration for CQR month intervals
      (b) isotonic-regression calibration for time-dependent horizon models

    Both use the same calibration partition to avoid a fourth split that
    would further reduce already-small event counts.
    """
    idx = np.arange(len(df), dtype=int)
    y   = df["OS_EVENT"].to_numpy(dtype=int)

    if len(np.unique(y)) < 2:
        raise ValueError("OS_EVENT has only one class; cannot stratify.")

    try:
        idx_train, idx_tmp = train_test_split(
            idx, test_size=0.4, random_state=RANDOM_STATE, stratify=y
        )
        idx_cal, idx_test = train_test_split(
            idx_tmp, test_size=0.5, random_state=RANDOM_STATE, stratify=y[idx_tmp]
        )
    except ValueError as exc:
        raise ValueError(
            "Stratified split failed — likely too few minority-class rows."
        ) from exc

    split_report: Dict[str, Dict[str, Any]] = {}
    for name, sidx in (("train", idx_train), ("calibration", idx_cal), ("test", idx_test)):
        nr = int(len(sidx))
        ne = int(y[sidx].sum())
        split_report[name] = {
            "rows": nr, "events": ne, "censored": nr - ne,
            "event_rate": float(ne / nr) if nr else float("nan"),
        }
        if ne == 0 or (nr - ne) == 0:
            raise ValueError(f"Split '{name}' has only one class.")

    return idx_train, idx_cal, idx_test, split_report


# ──────────────────────────────────────────────────────────────────────────────
# Feature column identification
# ──────────────────────────────────────────────────────────────────────────────

def get_feature_sets(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    clinical_candidates = ["AGE", "SEX", "RACE", "ETHNICITY", "CANCER_TYPE", "AGE_GROUP"]
    clinical_cols = [c for c in clinical_candidates if c in df.columns]
    expr_cols     = sorted(c for c in df.columns if c.startswith("EXPR_"))
    if not clinical_cols and not expr_cols:
        raise ValueError("No usable feature columns found.")
    return clinical_cols, expr_cols


# ──────────────────────────────────────────────────────────────────────────────
# Collinearity filtering helpers
# ──────────────────────────────────────────────────────────────────────────────

def _empty_drop_table() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "group", "feature_dropped", "anchor_feature",
        "abs_correlation", "threshold", "method",
    ])


def _drop_by_full_correlation(
    numeric_df: pd.DataFrame, threshold: float, group_name: str,
) -> Tuple[Set[str], pd.DataFrame, Dict[str, Any]]:
    """
    For every pair (i, j) with i < j and |corr| > threshold, mark j as
    redundant. We iterate over columns in order; the first column of a
    correlated pair is kept, the second is dropped. This is deterministic
    and equivalent to keeping a maximal independent set under the ordering.
    """
    abs_corr = numeric_df.corr(method="pearson").abs()
    if abs_corr.empty:
        return set(), _empty_drop_table(), {
            "method": "full_correlation_matrix", "n_pairs_above_threshold": 0
        }

    upper_mask = np.triu(np.ones(abs_corr.shape, dtype=bool), k=1)
    upper      = abs_corr.where(upper_mask)
    upper_arr  = upper.to_numpy(dtype=float)

    ri, ci = np.where(np.isfinite(upper_arr) & (upper_arr > threshold))
    high_pairs = sorted(
        [(str(upper.index[i]), str(upper.columns[j]), float(upper_arr[i, j]))
         for i, j in zip(ri, ci)],
        key=lambda t: t[2], reverse=True,
    )

    drop_rows: List[Dict[str, Any]] = []
    drop_set:  Set[str] = set()

    for col in upper.columns:
        high = upper[col][upper[col] > threshold].dropna()
        if high.empty:
            continue
        anchor   = str(high.idxmax())
        corr_val = float(high.max())
        drop_set.add(str(col))
        drop_rows.append({
            "group": group_name, "feature_dropped": str(col),
            "anchor_feature": anchor, "abs_correlation": corr_val,
            "threshold": float(threshold), "method": "full_correlation_matrix",
        })

    preview = [{"feature_a": a, "feature_b": b, "abs_correlation": c}
               for a, b, c in high_pairs[:20]]
    return drop_set, pd.DataFrame(drop_rows), {
        "method": "full_correlation_matrix",
        "n_pairs_above_threshold": int(len(high_pairs)),
        "top_high_correlation_pairs_preview": preview,
    }


def _drop_by_incremental_correlation(
    numeric_df: pd.DataFrame, threshold: float, group_name: str,
) -> Tuple[Set[str], pd.DataFrame, Dict[str, Any]]:
    """
    Memory-efficient O(n_kept × n_cols) pass for very wide matrices.
    For each new column j, compute its maximum abs Pearson correlation
    against all already-kept columns. Drop j if max > threshold.
    """
    cols = list(numeric_df.columns)
    if len(cols) < 2:
        return set(), _empty_drop_table(), {
            "method": "incremental_against_kept", "n_dropped": 0
        }

    X = np.asarray(numeric_df.values, dtype=float)
    # Median-impute NaN/Inf only for correlation arithmetic
    medians = np.nanmedian(X, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    bad     = ~np.isfinite(X)
    if np.any(bad):
        r, c = np.where(bad)
        X[r, c] = medians[c]
    X     = X - np.mean(X, axis=0, keepdims=True)
    norms = np.linalg.norm(X, axis=0)

    kept_idx:  List[int] = []
    drop_rows: List[Dict[str, Any]] = []

    for j, col_name in enumerate(cols):
        if not kept_idx:
            kept_idx.append(j)
            continue
        if norms[j] < 1e-12:   # constant column — keep but warn
            kept_idx.append(j)
            continue
        k     = np.asarray(kept_idx, dtype=int)
        valid = norms[k] >= 1e-12
        k     = k[valid]
        if len(k) == 0:
            kept_idx.append(j)
            continue
        corr_vec  = np.abs((X[:, k].T @ X[:, j]) / (norms[k] * norms[j]))
        best_idx  = int(np.argmax(corr_vec))
        best_corr = float(corr_vec[best_idx])
        if best_corr > threshold:
            drop_rows.append({
                "group": group_name, "feature_dropped": str(col_name),
                "anchor_feature": str(cols[int(k[best_idx])]),
                "abs_correlation": best_corr,
                "threshold": float(threshold),
                "method": "incremental_against_kept",
            })
        else:
            kept_idx.append(j)

    drop_set = {str(r["feature_dropped"]) for r in drop_rows}
    return drop_set, pd.DataFrame(drop_rows), {
        "method": "incremental_against_kept", "n_dropped": int(len(drop_rows))
    }


def drop_collinear_features(
    df_reference: pd.DataFrame,
    columns: List[str],
    threshold: float,
    group_name: str,
) -> Tuple[List[str], pd.DataFrame, Dict[str, Any]]:
    if not columns:
        return [], _empty_drop_table(), {
            "group": group_name, "threshold": float(threshold),
            "input_feature_count": 0, "numeric_evaluated_count": 0,
            "dropped_count": 0, "kept_count": 0,
            "skipped_non_numeric_or_too_sparse": [],
            "method_details": {"method": "none"},
        }

    numeric_data: Dict[str, pd.Series] = {}
    skipped_sparse:   List[str] = []
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
            "group": group_name, "threshold": float(threshold),
            "input_feature_count": int(len(columns)),
            "numeric_evaluated_count": int(numeric_df.shape[1]),
            "dropped_count": 0, "kept_count": int(len(columns)),
            "skipped_non_numeric_or_too_sparse": skipped_sparse,
            "skipped_constant_or_near_constant": skipped_constant,
            "method_details": {"method": "insufficient_numeric_features"},
        }

    if numeric_df.shape[1] <= COLLINEARITY_MAX_FULL_CORR_FEATURES:
        drop_set, drop_table, method_details = _drop_by_full_correlation(
            numeric_df, threshold, group_name
        )
    else:
        drop_set, drop_table, method_details = _drop_by_incremental_correlation(
            numeric_df, threshold, group_name
        )

    kept_columns = [c for c in columns if c not in drop_set]
    if drop_table.empty:
        drop_table = _empty_drop_table()

    summary = {
        "group": group_name, "threshold": float(threshold),
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
    """
    Fit collinearity filter on training rows ONLY (no look-ahead into
    calibration or test). Apply same column mask to calibration/test at
    transform time.
    """
    expr_kept, expr_drop_table, expr_summary = drop_collinear_features(
        df_reference=df_train, columns=expr_cols,
        threshold=threshold, group_name="expression",
    )

    # Clinical: only attempt filtering on numeric columns; non-numeric are
    # kept as-is since correlation is undefined for categoricals.
    clin_numeric: List[str] = []
    clin_non_numeric: List[str] = []
    for col in clinical_cols:
        if int(pd.to_numeric(df_train[col], errors="coerce").notna().sum()) >= 2:
            clin_numeric.append(col)
        else:
            clin_non_numeric.append(col)

    clin_num_kept, clin_drop_table, clin_summary = drop_collinear_features(
        df_reference=df_train, columns=clin_numeric,
        threshold=threshold, group_name="clinical_numeric",
    )
    clin_dropped = set(clin_summary.get("dropped_features", []))
    clinical_final = [c for c in clinical_cols if c not in clin_dropped]

    all_tables = [t for t in (expr_drop_table, clin_drop_table) if not t.empty]
    if all_tables:
        dropped_df = pd.concat(all_tables, axis=0, ignore_index=True).sort_values(
            ["group", "abs_correlation", "feature_dropped"],
            ascending=[True, False, True],
        )
    else:
        dropped_df = _empty_drop_table()

    summary = {
        "threshold_abs_pearson": float(threshold),
        "fit_reference": "train_split_only",
        "expression": expr_summary,
        "clinical_numeric": clin_summary,
        "clinical_non_numeric_or_sparse_kept_as_is": clin_non_numeric,
        "feature_counts": {
            "expr_before": int(len(expr_cols)),
            "expr_after":  int(len(expr_kept)),
            "clinical_before": int(len(clinical_cols)),
            "clinical_after":  int(len(clinical_final)),
            "total_before": int(len(expr_cols) + len(clinical_cols)),
            "total_after":  int(len(expr_kept) + len(clinical_final)),
            "total_dropped": int(
                (len(expr_cols) + len(clinical_cols))
                - (len(expr_kept) + len(clinical_final))
            ),
        },
    }
    return clinical_final, expr_kept, dropped_df, summary


# ──────────────────────────────────────────────────────────────────────────────
# Feature preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def _fit_clinical_block(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    clinical_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, Optional[ColumnTransformer], List[str], List[str]]:
    empty_tr = np.empty((len(df_tr), 0), dtype=float)
    empty_va = np.empty((len(df_va), 0), dtype=float)
    if not clinical_cols:
        return empty_tr, empty_va, None, [], []

    c_tr = df_tr[clinical_cols].copy()
    c_va = df_va[clinical_cols].copy()
    num_cols = c_tr.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in c_tr.columns if c not in num_cols]

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc",  StandardScaler())]),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh",  make_one_hot_encoder())]),
            cat_cols,
        ))
    if not transformers:
        return empty_tr, empty_va, None, [], []

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    Xc_tr = np.asarray(pre.fit_transform(c_tr), dtype=float)
    Xc_va = np.asarray(pre.transform(c_va),     dtype=float)
    return Xc_tr, Xc_va, pre, num_cols, cat_cols


def _fit_expression_block(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    expr_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, Optional[Pipeline], int]:
    empty_tr = np.empty((len(df_tr), 0), dtype=float)
    empty_va = np.empty((len(df_va), 0), dtype=float)
    if not expr_cols:
        return empty_tr, empty_va, None, 0

    e_tr = df_tr[expr_cols].copy()
    e_va = df_va[expr_cols].copy()
    n_comp = min(EXPR_PCA_COMPONENTS, int(e_tr.shape[0]), int(e_tr.shape[1]))
    if n_comp < 1:
        raise ValueError("Cannot configure PCA with fewer than 1 component.")

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=n_comp, random_state=RANDOM_STATE)),
    ])
    Xe_tr = np.asarray(pipe.fit_transform(e_tr), dtype=float)
    Xe_va = np.asarray(pipe.transform(e_va),     dtype=float)
    return Xe_tr, Xe_va, pipe, int(n_comp)


def fit_transform_features(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    clinical_cols: List[str],
    expr_cols: List[str],
) -> Tuple[
    np.ndarray, np.ndarray,
    Optional[ColumnTransformer], Optional[Pipeline],
    StandardScaler, Dict[str, Any],
]:
    Xc_tr, Xc_va, clin_pre, num_cols, cat_cols = _fit_clinical_block(
        df_tr, df_va, clinical_cols
    )
    Xe_tr, Xe_va, expr_pipe, n_comp = _fit_expression_block(
        df_tr, df_va, expr_cols
    )

    blocks_tr = [a for a in (Xc_tr, Xe_tr) if a.shape[1] > 0]
    blocks_va = [a for a in (Xc_va, Xe_va) if a.shape[1] > 0]
    if not blocks_tr:
        raise ValueError("No features remained after preprocessing.")

    X_tr = np.hstack(blocks_tr)
    X_va = np.hstack(blocks_va)
    if X_tr.shape[1] != X_va.shape[1]:
        raise ValueError("Feature dimension mismatch between train and validation.")

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_va   = scaler.transform(X_va)

    if not (np.isfinite(X_tr).all() and np.isfinite(X_va).all()):
        raise ValueError("Non-finite values after scaling.")

    stds = np.nanstd(X_tr, axis=0)
    checks: Dict[str, Any] = {
        "clinical_num_cols":         num_cols,
        "clinical_cat_cols":         cat_cols,
        "clinical_output_dim":       int(Xc_tr.shape[1]),
        "expr_input_dim":            int(len(expr_cols)),
        "expr_pca_components":       int(n_comp),
        "expr_output_dim":           int(Xe_tr.shape[1]),
        "final_dim":                 int(X_tr.shape[1]),
        "final_train_mean_abs_max":  float(np.max(np.abs(np.nanmean(X_tr, axis=0)))),
        "final_train_std_min":       float(np.min(stds)),
        "final_train_std_max":       float(np.max(stds)),
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
    X = np.hstack(blocks)
    X = scaler.transform(X)
    if not np.isfinite(X).all():
        raise ValueError("Non-finite values after transform.")
    return X


# ──────────────────────────────────────────────────────────────────────────────
# Cox CV hyperparameter search
# ──────────────────────────────────────────────────────────────────────────────

def run_cv_tuning(
    df_train: pd.DataFrame,
    clinical_cols: List[str],
    expr_cols: List[str],
) -> pd.DataFrame:
    """
    StratifiedKFold cross-validation over (alpha, l1_ratio) grid for Cox
    Elastic-Net. Metric: concordance index on validation fold.
    """
    y            = df_train["OS_EVENT"].to_numpy(dtype=int)
    class_counts = np.bincount(y, minlength=2)
    minority     = int(np.min(class_counts[class_counts > 0]))
    if len(class_counts[class_counts > 0]) < 2:
        raise ValueError("CV needs both classes.")
    n_splits = min(CV_FOLDS, minority)
    if n_splits < 2:
        raise ValueError(
            f"Too few minority-class samples ({minority}) for stratified CV."
        )

    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for cox_alpha in ALPHA_GRID:
        for l1_ratio in L1_GRID:
            fold_scores: List[float] = []
            failures:    List[str]   = []

            for fold_id, (tr_idx, va_idx) in enumerate(
                skf.split(np.zeros(len(df_train)), y), start=1
            ):
                tr = df_train.iloc[tr_idx]
                va = df_train.iloc[va_idx]
                try:
                    X_tr, X_va, _, _, _, _ = fit_transform_features(
                        tr, va, clinical_cols, expr_cols
                    )
                    t_tr = tr["OS_MONTHS"].to_numpy(dtype=float)
                    e_tr = tr["OS_EVENT"].to_numpy(dtype=int)
                    t_va = va["OS_MONTHS"].to_numpy(dtype=float)
                    e_va = va["OS_EVENT"].to_numpy(dtype=int)

                    mdl = CoxElasticNet(
                        alpha=cox_alpha, l1_ratio=l1_ratio, maxiter=MAXITER
                    )
                    mdl.fit(X_tr, t_tr, e_tr)
                    if not bool(getattr(mdl, "success_", True)):
                        failures.append(f"fold_{fold_id}:not_converged")
                        continue

                    risk = mdl.predict_risk(X_va)
                    ci   = concordance_index_censored(t_va, e_va, risk)
                    if not np.isfinite(ci):
                        failures.append(f"fold_{fold_id}:non_finite_ci")
                        continue
                    fold_scores.append(float(ci))
                except Exception as exc:
                    failures.append(f"fold_{fold_id}:{type(exc).__name__}")

            rows.append({
                "alpha":         float(cox_alpha),
                "l1_ratio":      float(l1_ratio),
                "n_valid_folds": int(len(fold_scores)),
                "n_failed_folds":int(n_splits - len(fold_scores)),
                "mean_c_index":  float(np.mean(fold_scores)) if fold_scores else float("nan"),
                "std_c_index":   float(np.std(fold_scores))  if fold_scores else float("nan"),
                "fold_scores":   ", ".join(f"{s:.4f}" for s in fold_scores),
                "failure_notes": "; ".join(failures),
            })

    cv_df = pd.DataFrame(rows)
    return cv_df.sort_values(
        ["n_valid_folds", "mean_c_index", "std_c_index"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def select_best_hyperparameters(cv_df: pd.DataFrame) -> pd.Series:
    viable = cv_df[
        np.isfinite(cv_df["mean_c_index"]) & (cv_df["n_valid_folds"] >= 2)
    ].copy()
    if viable.empty:
        raise ValueError("No viable hyperparameter combination found in CV.")
    return viable.sort_values(
        ["n_valid_folds", "mean_c_index", "std_c_index"],
        ascending=[False, False, True],
    ).iloc[0]


# ──────────────────────────────────────────────────────────────────────────────
# CQR month-interval prediction
# ──────────────────────────────────────────────────────────────────────────────

def cqr_month_interval(
    X_tr:  np.ndarray,
    t_tr:  np.ndarray,
    e_tr:  np.ndarray,
    X_cal: np.ndarray,
    t_cal: np.ndarray,
    e_cal: np.ndarray,
    X_te:  np.ndarray,
    t_te:  np.ndarray,
    e_te:  np.ndarray,
    alpha: float = CONFORMAL_ALPHA,
) -> Tuple[np.ndarray, np.ndarray, float, float, Dict[str, Any]]:
    """
    CQR survival-time intervals (Romano, Patterson & Candès, NeurIPS 2019).

    The output [lo, hi] answers: "In how many months is this patient likely
    to experience the event?" with approximately 90% marginal coverage for
    patients who experience the event.

    Algorithm
    ---------
    1. Select event-only rows from training set (OS_EVENT == 1).
       These are the only rows for which OS_MONTHS = true event time.
       Censored rows have OS_MONTHS = min(T_true, C) which is a lower bound
       on T_true, not T_true itself — using them as regression targets would
       systematically under-estimate quantiles.

    2. Fit two GradientBoostingRegressor models at quantile levels
       alpha_lo = alpha/2 and alpha_hi = 1 - alpha/2 on the raw month scale.
       Raw scale is used deliberately: no log transform → no asymmetric
       back-transformation → no exponential upper bound explosion.

    3. Compute CQR nonconformity scores on event-only calibration rows:
           E_i = max(q_lo(X_i) − Y_i,  Y_i − q_hi(X_i))
       Score is positive when Y_i falls outside [q_lo, q_hi], negative when
       inside. This is eq. 9 in Romano et al. 2019.

    4. Conservative conformal quantile:
           qhat = quantile(E, ceil((n+1)*(1-alpha)) / n)
       The ceiling formula (Lei et al. 2018) guarantees the finite-sample
       inequality P(Y ∈ interval) >= 1 − alpha under exchangeability.

    5. Final test-set intervals (all test rows, including censored):
           lo = max(0,  q_lo(X_te) − qhat)
           hi = max(lo, q_hi(X_te) + qhat)

    Coverage statement
    ------------------
    The interval is valid (marginal coverage ≥ 90%) for exchangeable draws
    from the event-only sub-population. Exchangeability holds when
    train/cal/test are random splits of an i.i.d. sample and we filter to
    the same event condition. Coverage for censored patients is not claimed
    because their true event time is unobserved.

    Parameters
    ----------
    X_tr, X_cal, X_te : pre-scaled feature matrices
    t_tr, t_cal, t_te : OS_MONTHS (raw months, positive floats)
    e_tr, e_cal, e_te : OS_EVENT (1=event, 0=censored)
    alpha             : miscoverage level

    Returns
    -------
    lo       : lower bounds for all test rows (nan if degenerate)
    hi       : upper bounds for all test rows (nan if degenerate)
    qhat     : conformal correction in months (nan if degenerate)
    coverage : empirical coverage on test event rows (nan if none)
    details  : audit dict
    """
    n_te = len(X_te)
    lo   = np.full(n_te, np.nan, dtype=float)
    hi   = np.full(n_te, np.nan, dtype=float)

    tr_mask  = e_tr  == 1
    cal_mask = e_cal == 1
    te_mask  = e_te  == 1

    n_tr_ev  = int(tr_mask.sum())
    n_cal_ev = int(cal_mask.sum())
    n_te_ev  = int(te_mask.sum())

    details: Dict[str, Any] = {
        "method":               "CQR_Romano_Patterson_Candes_NeurIPS2019",
        "reference":            "arXiv:1905.03222",
        "alpha":                float(alpha),
        "alpha_lo_quantile":    float(alpha / 2.0),
        "alpha_hi_quantile":    float(1.0 - alpha / 2.0),
        "n_train_events":       n_tr_ev,
        "n_cal_events":         n_cal_ev,
        "n_test_events":        n_te_ev,
        "cqr_min_train_events": CQR_MIN_TRAIN_EVENTS,
        "cqr_min_cal_events":   CQR_MIN_CAL_EVENTS,
        "coverage_claimed_for": "event-only_test_rows_under_exchangeability",
        "coverage_not_claimed_for": "censored_test_rows",
        "status":               "ok",
    }

    # ── Guard: insufficient training events ──────────────────────────────
    if n_tr_ev < CQR_MIN_TRAIN_EVENTS:
        details["status"] = (
            f"skipped: only {n_tr_ev} training events "
            f"(need >= {CQR_MIN_TRAIN_EVENTS})"
        )
        return lo, hi, np.nan, np.nan, details

    # ── Guard: insufficient calibration events ────────────────────────────
    if n_cal_ev < CQR_MIN_CAL_EVENTS:
        details["status"] = (
            f"skipped: only {n_cal_ev} calibration events "
            f"(need >= {CQR_MIN_CAL_EVENTS})"
        )
        return lo, hi, np.nan, np.nan, details

    X_tr_ev  = X_tr[tr_mask]
    y_tr_ev  = t_tr[tr_mask]     # true event times in raw months
    X_cal_ev = X_cal[cal_mask]
    y_cal_ev = t_cal[cal_mask]

    # ── Step 1: Fit two quantile GBR models ───────────────────────────────
    alpha_lo = alpha / 2.0          # 0.05 for 90% coverage
    alpha_hi = 1.0 - alpha / 2.0   # 0.95
    try:
        qr_lo = GradientBoostingRegressor(alpha=alpha_lo, **CQR_GBR_PARAMS)
        qr_hi = GradientBoostingRegressor(alpha=alpha_hi, **CQR_GBR_PARAMS)
        qr_lo.fit(X_tr_ev, y_tr_ev)
        qr_hi.fit(X_tr_ev, y_tr_ev)
    except Exception as exc:
        details["status"] = f"skipped: QR fit failed ({type(exc).__name__}: {exc})"
        return lo, hi, np.nan, np.nan, details

    # ── Step 2: Nonconformity scores on calibration events ─────────────────
    q_lo_cal = qr_lo.predict(X_cal_ev)
    q_hi_cal = qr_hi.predict(X_cal_ev)
    # Enforce q_lo ≤ q_hi (quantile crossing can occur with independent models)
    q_lo_cal = np.minimum(q_lo_cal, q_hi_cal)
    q_hi_cal = np.maximum(q_lo_cal, q_hi_cal)  # intentional re-read after update

    # CQR nonconformity score (Romano et al. 2019, eq. 9):
    #   positive  → Y outside the raw QR interval  (model under-covered here)
    #   negative  → Y inside the raw QR interval   (model over-covered here)
    scores = np.maximum(q_lo_cal - y_cal_ev, y_cal_ev - q_hi_cal)

    details["calibration_score_min"]    = float(np.min(scores))
    details["calibration_score_median"] = float(np.median(scores))
    details["calibration_score_p90"]    = float(np.quantile(scores, 0.90))
    details["calibration_score_max"]    = float(np.max(scores))
    details["calibration_score_mean"]   = float(np.mean(scores))
    details["raw_qr_interval_width_mean_cal"] = float(
        np.mean(q_hi_cal - q_lo_cal)
    )

    # ── Step 3: Conservative conformal quantile ────────────────────────────
    qhat = conformal_quantile(scores, alpha)
    details["qhat_months"] = float(qhat)

    if not np.isfinite(qhat):
        details["status"] = (
            "skipped: qhat is +inf — calibration set too small for "
            f"requested alpha={alpha} with n_cal_events={n_cal_ev}"
        )
        return lo, hi, np.nan, np.nan, details

    # ── Step 4: Predict intervals for all test rows ────────────────────────
    q_lo_te = qr_lo.predict(X_te)
    q_hi_te = qr_hi.predict(X_te)
    q_lo_te = np.minimum(q_lo_te, q_hi_te)
    q_hi_te = np.maximum(q_lo_te, q_hi_te)   # re-read after update

    lo = np.clip(q_lo_te - qhat, 0.0, None)
    hi = q_hi_te + qhat
    hi = np.maximum(lo, hi)  # ensure hi >= lo after lo clipping

    details["mean_interval_width_months_all_test"] = float(
        np.mean(hi - lo)
    )
    details["median_interval_width_months_all_test"] = float(
        np.median(hi - lo)
    )

    # ── Step 5: Empirical coverage on test event rows ──────────────────────
    if n_te_ev > 0:
        lo_ev = lo[te_mask]
        hi_ev = hi[te_mask]
        y_te_ev = t_te[te_mask]
        coverage = float(np.mean((y_te_ev >= lo_ev) & (y_te_ev <= hi_ev)))
        details["coverage_test_events_only"]             = coverage
        details["mean_interval_width_months_test_events"] = float(
            np.mean(hi_ev - lo_ev)
        )
        details["median_interval_width_months_test_events"] = float(
            np.median(hi_ev - lo_ev)
        )
    else:
        coverage = np.nan
        details["coverage_test_events_only"] = float("nan")

    return lo, hi, float(qhat), coverage, details


# ──────────────────────────────────────────────────────────────────────────────
# Time-dependent conformal (horizon binary labels)
# ──────────────────────────────────────────────────────────────────────────────

def _horizon_labels(
    time_arr:  np.ndarray,
    event_arr: np.ndarray,
    horizon:   float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign binary labels relative to a fixed time horizon h:
      y = 1  if patient had confirmed event before h  (event=1 AND time <= h)
      y = 0  if patient is confirmed event-free at h   (time > h, any event)
    Patients censored before h are excluded (status at h is unknown).

    The keep mask selects the rows with known horizon-status; the returned
    y array is indexed relative to those rows.
    """
    confirmed_event  = (event_arr == 1) & (time_arr <= horizon)
    confirmed_no_evt = time_arr > horizon
    keep = confirmed_event | confirmed_no_evt
    y    = np.where(confirmed_event[keep], 1, 0)
    return keep, y


def time_dependent_conformal(
    r_cal: np.ndarray,
    t_cal: np.ndarray,
    e_cal: np.ndarray,
    r_te:  np.ndarray,
    t_te:  np.ndarray,
    e_te:  np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, Dict[str, Any]]]:
    """
    For each horizon h in HORIZONS_MONTHS:
      1. Derive binary "event by horizon h" labels for calibration and test
         (excluding patients censored before h — their status is unknown).
      2. Fit IsotonicRegression to map Cox risk score → P(event by h).
         IsotonicRegression is monotone-increasing by default (default), which
         is correct: higher risk score → higher event probability.
      3. Apply split conformal with score = |y - p̂| (bounded in [0,1])
         using the same conformal_quantile() as CQR.
      4. Produce [p_lo, p_hi] probability intervals clipped to [0, 1].
      5. Select classification threshold via Youden J on calibration labels.

    Limitation: IsotonicRegression is fit on (r_cal, y_cal) where r_cal is
    the Cox risk score (a score, not a probability). IsotonicRegression maps
    it monotonically to [0,1] via PAVA. This is a non-parametric calibration
    step, not a model with a generative story. It can overfit with small n.
    """
    rows_metrics: List[Dict[str, Any]] = []
    rows_pred:    List[Dict[str, Any]] = []
    rows_cls:     List[Dict[str, Any]] = []
    calibrators:  Dict[int, Dict[str, Any]] = {}

    for h in HORIZONS_MONTHS:
        keep_cal, y_cal = _horizon_labels(t_cal, e_cal, h)
        keep_te,  y_te  = _horizon_labels(t_te,  e_te,  h)
        n_cal = int(len(y_cal))
        n_te  = int(len(y_te))

        base_row: Dict[str, Any] = {
            "horizon_months": h,
            "n_cal_known": n_cal,
            "n_test_known": n_te,
        }

        if n_cal == 0 or n_te == 0:
            rows_metrics.append({**base_row, "note": "no_known_labels"})
            continue
        if len(np.unique(y_cal)) < 2:
            rows_metrics.append({**base_row, "note": "insufficient_cal_class_variation"})
            continue
        if len(np.unique(y_te)) < 2:
            rows_metrics.append({**base_row, "note": "insufficient_test_class_variation"})
            continue

        # IsotonicRegression: increasing=True (default) is correct here.
        # Higher Cox risk score should monotonically map to higher event prob.
        try:
            iso = IsotonicRegression(out_of_bounds="clip")  # increasing=True default
            p_cal = iso.fit_transform(r_cal[keep_cal], y_cal)
            p_te  = iso.predict(r_te[keep_te])
        except Exception as exc:
            rows_metrics.append({**base_row, "note": f"isotonic_failed:{type(exc).__name__}"})
            continue

        if not (np.isfinite(p_cal).all() and np.isfinite(p_te).all()):
            rows_metrics.append({**base_row, "note": "non_finite_isotonic_probs"})
            continue

        # Split conformal on absolute residuals from {0,1} labels
        scores  = np.abs(y_cal - p_cal)
        qhat_h  = conformal_quantile(scores, CONFORMAL_ALPHA)
        if not np.isfinite(qhat_h):
            rows_metrics.append({**base_row, "note": "non_finite_conformal_quantile"})
            continue

        p_lo     = np.clip(p_te - qhat_h, 0.0, 1.0)
        p_hi     = np.clip(p_te + qhat_h, 0.0, 1.0)
        coverage = float(np.mean((y_te >= p_lo) & (y_te <= p_hi)))

        rows_metrics.append({
            **base_row,
            "qhat":               float(qhat_h),
            "coverage_known_test": coverage,
            "mean_interval_width": float(np.mean(p_hi - p_lo)),
            "note":               "ok",
        })

        # Youden J threshold from calibration labels only
        thr_grid = np.linspace(0, 1, 201)
        best_thr = 0.5
        best_j   = -1.0
        for thr in thr_grid:
            yhat_c              = (p_cal >= thr).astype(int)
            tn_c, fp_c, fn_c, tp_c = confusion_matrix(
                y_cal, yhat_c, labels=[0, 1]
            ).ravel()
            tpr = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
            tnr = tn_c / (tn_c + fp_c) if (tn_c + fp_c) > 0 else 0.0
            j   = tpr + tnr - 1.0
            if j > best_j:
                best_j   = j
                best_thr = float(thr)

        yhat_t               = (p_te >= best_thr).astype(int)
        tn, fp, fn, tp       = confusion_matrix(y_te, yhat_t, labels=[0, 1]).ravel()
        specificity          = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

        try:    roc_auc = float(roc_auc_score(y_te, p_te))
        except: roc_auc = float("nan")
        try:    pr_auc  = float(average_precision_score(y_te, p_te))
        except: pr_auc  = float("nan")

        rows_cls.append({
            "horizon_months":              h,
            "n_test_known":                n_te,
            "event_rate_test_known":       float(np.mean(y_te)),
            "threshold_from_calibration":  best_thr,
            "roc_auc":                     roc_auc,
            "pr_auc":                      pr_auc,
            "brier":                       float(brier_score_loss(y_te, p_te)),
            "accuracy":                    float(accuracy_score(y_te, yhat_t)),
            "precision":                   float(precision_score(y_te, yhat_t, zero_division=0)),
            "recall_sensitivity":          float(recall_score(y_te, yhat_t, zero_division=0)),
            "specificity":                 float(specificity),
            "f1":                          float(f1_score(y_te, yhat_t, zero_division=0)),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        })

        calibrators[int(h)] = {
            "isotonic":  iso,
            "qhat":      float(qhat_h),
            "threshold": float(best_thr),
        }

        for i_local, i_global in enumerate(np.where(keep_te)[0]):
            rows_pred.append({
                "horizon_months":                h,
                "test_row_index_within_split":   int(i_global),
                "risk_score":                    float(r_te[i_global]),
                "y_true_by_horizon":             int(y_te[i_local]),
                "p_event_hat":                   float(p_te[i_local]),
                "p_event_lo":                    float(p_lo[i_local]),
                "p_event_hi":                    float(p_hi[i_local]),
            })

    return (
        pd.DataFrame(rows_metrics),
        pd.DataFrame(rows_pred),
        pd.DataFrame(rows_cls),
        calibrators,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & clean ──────────────────────────────────────────────────
    df           = prepare_dataframe().reset_index(drop=True)
    df_no, outlier_report = remove_outliers_iqr(df)

    # ── 2. Split ─────────────────────────────────────────────────────────
    idx_train, idx_cal, idx_test, split_report = split_three_way(df_no)
    clinical_cols, expr_cols = get_feature_sets(df_no)

    # ── 3. Collinearity filter (fit on train only) ────────────────────────
    df_train_for_filter = df_no.iloc[idx_train].copy().reset_index(drop=True)
    clinical_cols_f, expr_cols_f, dropped_df, coll_summary = apply_collinearity_filter(
        df_train_for_filter, clinical_cols, expr_cols,
        threshold=COLLINEARITY_THRESHOLD,
    )
    dropped_df.to_csv(OUT_DIR / "collinearity_dropped_features.csv", index=False)
    (OUT_DIR / "collinearity_summary.json").write_text(
        json.dumps(coll_summary, indent=2), encoding="utf-8"
    )
    if not clinical_cols_f and not expr_cols_f:
        raise ValueError("All features removed by collinearity filter.")

    # ── 4. Leakage guard ──────────────────────────────────────────────────
    # Explicitly check that no outcome-derived column is in the feature set.
    selected = set(clinical_cols_f) | set(expr_cols_f)
    forbidden_exact    = {"OS_MONTHS", "OS_EVENT", "OS_STATUS",
                          "DFS_STATUS", "DFS_EVENT"}
    forbidden_prefixes = ("OS_", "DFS_")
    leaking = sorted(
        c for c in selected
        if c in forbidden_exact or any(c.startswith(p) for p in forbidden_prefixes)
    )
    if leaking:
        raise ValueError(f"Target-leakage features in model inputs: {leaking}")

    # ── 5. Cox CV tuning ──────────────────────────────────────────────────
    df_train = df_no.iloc[idx_train].copy().reset_index(drop=True)
    cv_df    = run_cv_tuning(df_train, clinical_cols_f, expr_cols_f)
    cv_df.to_csv(OUT_DIR / "hyperparameter_cv_results.csv", index=False)

    best      = select_best_hyperparameters(cv_df)
    best_cox_alpha = float(best["alpha"])
    best_l1        = float(best["l1_ratio"])

    # ── 6. Final feature transforms ───────────────────────────────────────
    tr  = df_no.iloc[idx_train].copy()
    cal = df_no.iloc[idx_cal].copy()
    te  = df_no.iloc[idx_test].copy()

    X_tr, X_cal, clin_pre, expr_pipe, scaler, feat_checks = fit_transform_features(
        tr, cal, clinical_cols_f, expr_cols_f
    )
    X_te = transform_features(
        te, clinical_cols_f, expr_cols_f, clin_pre, expr_pipe, scaler
    )

    t_tr  = tr["OS_MONTHS"].to_numpy(dtype=float)
    e_tr  = tr["OS_EVENT"].to_numpy(dtype=int)
    t_cal = cal["OS_MONTHS"].to_numpy(dtype=float)
    e_cal = cal["OS_EVENT"].to_numpy(dtype=int)
    t_te  = te["OS_MONTHS"].to_numpy(dtype=float)
    e_te  = te["OS_EVENT"].to_numpy(dtype=int)

    # ── 7. Fit final Cox model ────────────────────────────────────────────
    model = CoxElasticNet(
        alpha=best_cox_alpha, l1_ratio=best_l1, maxiter=MAXITER
    )
    model.fit(X_tr, t_tr, e_tr)
    if not bool(getattr(model, "success_", True)):
        raise RuntimeError(
            f"Cox model did not converge: {getattr(model, 'message_', 'unknown')}"
        )

    r_tr  = model.predict_risk(X_tr)
    r_cal = model.predict_risk(X_cal)
    r_te  = model.predict_risk(X_te)

    ci_train = float(concordance_index_censored(t_tr, e_tr, r_tr))
    ci_cal   = float(concordance_index_censored(t_cal, e_cal, r_cal))
    ci_test  = float(concordance_index_censored(t_te, e_te, r_te))

    # ── 8. CQR month intervals ────────────────────────────────────────────
    # Quantile models are fit on X_tr (event-only rows selected inside).
    # Conformal calibration uses X_cal (event-only rows selected inside).
    # Intervals produced for ALL X_te rows (censored get interval too, but
    # coverage guarantee applies only to event rows — see module docstring).
    lo, hi, qhat, coverage_ev, cqr_details = cqr_month_interval(
        X_tr, t_tr, e_tr,
        X_cal, t_cal, e_cal,
        X_te,  t_te,  e_te,
        alpha=CONFORMAL_ALPHA,
    )

    # Reference the fitted QR models for pickling (may not exist if skipped)
    qr_lo_model = None
    qr_hi_model = None
    if cqr_details["status"] == "ok":
        # The models are local to cqr_month_interval; re-expose via re-call
        # would be wasteful — instead capture via explicit return extension.
        # For now we note in the artifact that they must be re-derived.
        # TODO: refactor cqr_month_interval to also return the fitted models.
        pass

    # ── 9. Time-dependent conformal ───────────────────────────────────────
    td_metrics, td_preds, td_cls, td_calibrators = time_dependent_conformal(
        r_cal, t_cal, e_cal,
        r_te,  t_te,  e_te,
    )
    td_metrics.to_csv(OUT_DIR / "time_dependent_conformal_metrics.csv",       index=False)
    td_preds.to_csv(  OUT_DIR / "time_dependent_conformal_test_predictions.csv", index=False)
    td_cls.to_csv(    OUT_DIR / "time_dependent_horizon_classification_metrics.csv", index=False)

    # ── 10. Predictions CSV ───────────────────────────────────────────────
    patient_ids = pd.concat(
        [tr["PATIENT_ID"], cal["PATIENT_ID"], te["PATIENT_ID"]], axis=0
    ).astype(str).values

    pred_df = pd.DataFrame({
        "split":      (["train"] * len(tr) + ["calibration"] * len(cal)
                       + ["test"] * len(te)),
        "PATIENT_ID": patient_ids,
        "OS_MONTHS":  np.concatenate([t_tr, t_cal, t_te]),
        "OS_EVENT":   np.concatenate([e_tr, e_cal, e_te]),
        "risk_score": np.concatenate([r_tr, r_cal, r_te]),
        # lo / hi only populated for test; NaN for train and calibration.
        "pred_survival_lo_months_90": np.concatenate([
            np.full(len(tr),  np.nan),
            np.full(len(cal), np.nan),
            lo,
        ]),
        "pred_survival_hi_months_90": np.concatenate([
            np.full(len(tr),  np.nan),
            np.full(len(cal), np.nan),
            hi,
        ]),
    })
    pred_df.to_csv(OUT_DIR / "tuned_model_predictions.csv", index=False)

    coef_df = pd.DataFrame({
        "coef_index": np.arange(len(model.coef_)),
        "coef_value": model.coef_,
    }).sort_values("coef_value", key=np.abs, ascending=False)
    coef_df.to_csv(OUT_DIR / "tuned_model_coefficients.csv", index=False)

    # ── 11. Metrics JSON ──────────────────────────────────────────────────
    finite_widths    = (hi - lo)[np.isfinite(hi - lo)]
    mean_width_all   = float(np.mean(finite_widths))   if len(finite_widths) > 0 else float("nan")
    median_width_all = float(np.median(finite_widths)) if len(finite_widths) > 0 else float("nan")

    metrics: Dict[str, Any] = {
        "input_file": str(INPUT_PATH),
        "best_cox_params": {"alpha": best_cox_alpha, "l1_ratio": best_l1},
        "cv_best_mean_c_index":  float(best["mean_c_index"]),
        "cv_best_std_c_index":   float(best["std_c_index"]),
        "cv_best_valid_folds":   int(best["n_valid_folds"]),
        "n_total_after_outlier_filter": int(len(df_no)),
        "n_train":       int(len(tr)),
        "n_calibration": int(len(cal)),
        "n_test":        int(len(te)),
        "events_train":       int(e_tr.sum()),
        "events_calibration": int(e_cal.sum()),
        "events_test":        int(e_te.sum()),
        "cox_optimizer_success":    bool(getattr(model, "success_", True)),
        "cox_optimizer_message":    str(getattr(model,  "message_", "unknown")),
        "cox_optimizer_iterations": int(getattr(model,  "n_iter_",  -1)),
        "c_index_train":       ci_train,
        "c_index_calibration": ci_cal,
        "c_index_test":        ci_test,
        "collinearity_filter": {
            "threshold_abs_pearson": COLLINEARITY_THRESHOLD,
            "clinical_before": int(len(clinical_cols)),
            "clinical_after":  int(len(clinical_cols_f)),
            "expr_before":     int(len(expr_cols)),
            "expr_after":      int(len(expr_cols_f)),
            "total_dropped":   int(
                (len(clinical_cols) + len(expr_cols))
                - (len(clinical_cols_f) + len(expr_cols_f))
            ),
        },
        "cqr_month_interval": {
            **cqr_details,
            "mean_interval_width_months_all_test":    mean_width_all,
            "median_interval_width_months_all_test":  median_width_all,
        },
        "time_dependent_conformal": {
            "horizons_months":          HORIZONS_MONTHS,
            "n_horizons_reported":      int(td_metrics.shape[0]),
            "n_horizons_with_models":   int(len(td_calibrators)),
        },
    }
    (OUT_DIR / "tuned_model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # ── 12. Consistency / leakage audit JSON ──────────────────────────────
    consistency: Dict[str, Any] = {
        "input_file":                    str(INPUT_PATH),
        "rows_before_outlier_filter":    int(len(df)),
        "rows_after_outlier_filter":     int(len(df_no)),
        "outlier_report":                outlier_report,
        "split_report":                  split_report,
        "feature_checks":                feat_checks,
        "feature_counts": {
            "clinical_cols_before_collinearity": int(len(clinical_cols)),
            "expr_cols_before_collinearity":     int(len(expr_cols)),
            "clinical_cols_after_collinearity":  int(len(clinical_cols_f)),
            "expr_cols_after_collinearity":      int(len(expr_cols_f)),
        },
        "collinearity_summary": coll_summary,
        "leakage_check_passed": True,
    }
    (OUT_DIR / "consistency_checks.json").write_text(
        json.dumps(consistency, indent=2), encoding="utf-8"
    )

    # ── 13. Serialise model artifact ──────────────────────────────────────
    artifact: Dict[str, Any] = {
        # Feature column lists
        "clinical_cols_before_collinearity": clinical_cols,
        "expr_cols_before_collinearity":     expr_cols,
        "clinical_cols_after_collinearity":  clinical_cols_f,
        "expr_cols_after_collinearity":      expr_cols_f,
        # Preprocessing objects (fit on training data only)
        "clin_pre":    clin_pre,
        "expr_pipe":   expr_pipe,
        "scaler":      scaler,
        # Cox model
        "cox_model":   model,
        "best_cox_alpha":   best_cox_alpha,
        "best_cox_l1_ratio": best_l1,
        # CQR details (QR models not stored — see TODO in step 8)
        "cqr_qhat":    qhat,
        "cqr_alpha":   CONFORMAL_ALPHA,
        "cqr_details": cqr_details,
        # Time-dependent calibrators
        "time_dependent_calibrators": td_calibrators,
        "horizons_months":            HORIZONS_MONTHS,
    }
    with open(OUT_DIR / "final_locked_model.pkl", "wb") as f:
        pickle.dump(artifact, f)

    # ── 14. README ────────────────────────────────────────────────────────
    ok  = cqr_details["status"] == "ok"
    readme_lines = [
        "# Cox ENet + CQR conformal survival intervals",
        "",
        "## Method summary",
        "Conformal intervals use Conformalized Quantile Regression (CQR).",
        "Reference: Romano, Patterson & Candès, NeurIPS 2019 (arXiv:1905.03222).",
        "",
        "CQR fits two GradientBoostingRegressor quantile models at levels",
        f"α/2 = {CONFORMAL_ALPHA/2} and 1-α/2 = {1-CONFORMAL_ALPHA/2} on",
        "event-only training rows (raw month scale, no log transform).",
        "Conformal correction qhat is computed on event-only calibration rows.",
        "Final interval: [q_lo(X) - qhat, q_hi(X) + qhat], clipped to [0, ∞).",
        "",
        "## Coverage statement",
        "~90% marginal coverage claimed for event-only test rows under",
        "exchangeability. Coverage for censored patients is NOT claimed.",
        "",
        "## Results",
        f"Input               : {INPUT_PATH}",
        f"Collinearity thresh : {COLLINEARITY_THRESHOLD}",
        f"Best Cox alpha      : {best_cox_alpha}",
        f"Best Cox l1_ratio   : {best_l1}",
        f"CV mean c-index     : {float(best['mean_c_index']):.4f}",
        f"Test c-index        : {ci_test:.4f}",
        f"CQR status          : {cqr_details['status']}",
        (f"CQR qhat (months)   : {qhat:.2f}" if ok and np.isfinite(qhat)
         else "CQR qhat            : N/A"),
        (f"CQR coverage (ev.)  : {coverage_ev:.4f}" if ok and np.isfinite(coverage_ev)
         else "CQR coverage        : N/A"),
        (f"Mean width (all te) : {mean_width_all:.1f} months" if np.isfinite(mean_width_all)
         else "Mean width          : N/A"),
        "",
        "## Wide interval diagnosis",
        "If intervals are wide, likely causes (in order of probability):",
        "  1. Low event rate -> few calibration events -> large qhat.",
        "  2. High genuine variability in OS_MONTHS (e.g. from longitudinal",
        "     model transformation) -> wide raw QR bands.",
        "  3. Small sample size.",
        "There is no algorithmic fix for (2) and (3) without more data.",
        "",
        "## Output files",
        "- collinearity_dropped_features.csv",
        "- collinearity_summary.json",
        "- hyperparameter_cv_results.csv",
        "- tuned_model_metrics.json",
        "- tuned_model_predictions.csv",
        "- tuned_model_coefficients.csv",
        "- time_dependent_conformal_metrics.csv",
        "- time_dependent_conformal_test_predictions.csv",
        "- time_dependent_horizon_classification_metrics.csv",
        "- consistency_checks.json",
        "- final_locked_model.pkl",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    print("Done.")
    print(f"Results written to: {OUT_DIR}")


if __name__ == "__main__":
    main()