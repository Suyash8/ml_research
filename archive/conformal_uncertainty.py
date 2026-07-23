"""
conformal_uncertainty.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Conformal survival-time intervals with high-uncertainty patient flagging.

PURPOSE
-------
This module augments the CQR output from the main pipeline with a structured
uncertainty analysis. For every test patient the conformal interval [lo, hi]
is computed as before, but additionally each patient is assessed along three
independent uncertainty axes:

  TYPE 1 — MODEL_UNCERTAIN (wide raw QR band)
    The GBR quantile models themselves disagree: q_hi − q_lo > threshold.
    This reflects aleatoric uncertainty — the model cannot pin down the
    event time regardless of conformal correction.
    Threshold: top quartile of (q_hi − q_lo) across test patients.

  TYPE 2 — RISK_AMBIGUOUS (mid-range Cox score)
    The Cox risk score is near the median, where small perturbations flip
    the ranking. The model has the least discriminative power here.
    Threshold: risk score in the interquartile range [Q25, Q75] of test
    risk scores (the "uncertain middle").

  TYPE 3 — UPPER_TAIL_UNCERTAIN (long predicted survival, high hi bound)
    hi > threshold, meaning the model predicts the patient may survive a
    long time but the upper bound is poorly constrained.
    Threshold: top quartile of hi across test patients.

A patient is HIGH_UNCERTAINTY if they trigger ANY of the three types.
A patient is VERY_HIGH_UNCERTAINTY if they trigger TWO or more types.

OUTPUT COLUMNS (added to predictions CSV)
-----------------------------------------
  qr_lo_raw            : q_lo(X) before conformal correction (raw months)
  qr_hi_raw            : q_hi(X) before conformal correction (raw months)
  qr_width_raw         : q_hi_raw − q_lo_raw (model's own uncertainty)
  conformal_lo         : lo = max(0, q_lo_raw − qhat)
  conformal_hi         : hi = q_hi_raw + qhat
  conformal_width      : hi − lo
  type1_model_uncertain: bool — raw QR width > Q75 of test widths
  type2_risk_ambiguous : bool — risk score in IQR of test risk scores
  type3_upper_tail     : bool — conformal_hi > Q75 of test hi values
  n_uncertainty_types  : int  — count of triggered types (0, 1, 2, or 3)
  high_uncertainty     : bool — n_uncertainty_types >= 1
  very_high_uncertainty: bool — n_uncertainty_types >= 2
  uncertainty_label    : human-readable comma-separated list of triggered types
  uncertainty_rank     : rank 1 = most uncertain patient (by n_types, then width)

DESIGN NOTES
------------
• Conformal is applied to ALL test patients — the uncertainty flags are
  purely additive metadata, not a filter on who gets an interval.
• Thresholds (Q75) are computed on test patients only, so they reflect
  the relative uncertainty within the cohort, not an absolute cutoff.
• To change thresholds, edit UNCERTAINTY_QUANTILE (default 0.75).
• The conformal guarantee (≥90% marginal coverage for event patients) is
  unaffected by the flagging.

References
----------
Romano Y, Patterson E, Candès EJ (2019). Conformalized Quantile Regression.
  NeurIPS 32. arXiv:1905.03222.
Lei J, G'Sell M, Rinaldo A, Tibshirani RJ, Wasserman L (2018).
  Distribution-free predictive inference for regression. JASA 113:1094-1111.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from tune_train_cox_enet_cqr_fixed import CoxElasticNet

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Paths — edit to match your project layout
BASE        = Path("d:/ml_research")
INPUT_PATH  = BASE / "data" / "preprocessed_cleaned" / "patient_multiomic_cleaned.parquet"
MODEL_PKL   = (
    BASE / "data" / "model_outputs" / "cox_enet_cqr_fixed" / "final_locked_model.pkl"
)
PRED_CSV    = (
    BASE / "data" / "model_outputs" / "cox_enet_cqr_fixed" / "tuned_model_predictions.csv"
)
OUT_DIR     = BASE / "data" / "model_outputs" / "conformal_uncertainty"

# Conformal settings
CONFORMAL_ALPHA   = 0.10   # 1 − CONFORMAL_ALPHA = 90% coverage target
CQR_MIN_TRAIN_EVENTS = 20
CQR_MIN_CAL_EVENTS   = 10

# Uncertainty flagging threshold — patients above this quantile on each
# axis are flagged for that uncertainty type.
UNCERTAINTY_QUANTILE = 0.75   # top 25% on each axis → flagged

# Hard cap on interval width (None = no cap).
# Set to e.g. 48.0 to cap at 4 years.
MAX_INTERVAL_WIDTH_MONTHS: Optional[float] = None

# CQR quantile regressor hyperparameters
CQR_GBR_PARAMS: Dict[str, Any] = {
    "loss":              "quantile",
    "n_estimators":      300,
    "max_depth":         3,
    "learning_rate":     0.04,
    "min_samples_leaf":  10,
    "min_samples_split": 10,
    "subsample":         0.8,
    "random_state":      42,
}


# ─────────────────────────────────────────────────────────────────────────────
# Conformal quantile  (Lei et al. 2018)
# ─────────────────────────────────────────────────────────────────────────────

def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Conservative finite-sample (1−α) quantile.
    level = ceil((n+1)*(1−α)) / n
    Returns +inf when the calibration set is too small.
    """
    n = int(len(scores))
    if n == 0:
        return np.inf
    level = float(np.ceil((n + 1.0) * (1.0 - alpha))) / n
    return float(np.quantile(scores, level)) if level <= 1.0 else np.inf


# ─────────────────────────────────────────────────────────────────────────────
# CQR interval computation
# ─────────────────────────────────────────────────────────────────────────────

def fit_cqr_models(
    X_tr_ev: np.ndarray,
    y_tr_ev: np.ndarray,
    alpha: float,
) -> Tuple[GradientBoostingRegressor, GradientBoostingRegressor]:
    """
    Fit lower (α/2) and upper (1−α/2) quantile regressors on event-only
    training rows. Operates on raw months — no log transform.
    """
    alpha_lo = alpha / 2.0
    alpha_hi = 1.0 - alpha / 2.0
    qr_lo = GradientBoostingRegressor(alpha=alpha_lo, **CQR_GBR_PARAMS)
    qr_hi = GradientBoostingRegressor(alpha=alpha_hi, **CQR_GBR_PARAMS)
    qr_lo.fit(X_tr_ev, y_tr_ev)
    qr_hi.fit(X_tr_ev, y_tr_ev)
    return qr_lo, qr_hi


def compute_cqr_scores(
    qr_lo: GradientBoostingRegressor,
    qr_hi: GradientBoostingRegressor,
    X_cal_ev: np.ndarray,
    y_cal_ev: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute CQR nonconformity scores on calibration event rows.
    E_i = max(q_lo(X_i) − Y_i,  Y_i − q_hi(X_i))   [Romano 2019 eq. 9]
    Returns (q_lo_cal, q_hi_cal, scores).
    """
    q_lo_cal = qr_lo.predict(X_cal_ev)
    q_hi_cal = qr_hi.predict(X_cal_ev)
    # Quantile-crossing safeguard
    q_lo_cal = np.minimum(q_lo_cal, q_hi_cal)
    q_hi_cal = np.maximum(q_lo_cal, q_hi_cal)
    scores = np.maximum(q_lo_cal - y_cal_ev, y_cal_ev - q_hi_cal)
    return q_lo_cal, q_hi_cal, scores


def apply_cqr_to_test(
    qr_lo: GradientBoostingRegressor,
    qr_hi: GradientBoostingRegressor,
    X_te: np.ndarray,
    qhat: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce raw QR predictions and conformal intervals for all test rows.

    Returns
    -------
    q_lo_raw : raw lower quantile (before conformal correction)
    q_hi_raw : raw upper quantile (before conformal correction)
    lo       : conformal lower bound = max(0, q_lo_raw − qhat)
    hi       : conformal upper bound = max(lo, q_hi_raw + qhat)
    """
    q_lo_raw = qr_lo.predict(X_te)
    q_hi_raw = qr_hi.predict(X_te)
    # Quantile-crossing safeguard
    q_lo_raw = np.minimum(q_lo_raw, q_hi_raw)
    q_hi_raw = np.maximum(q_lo_raw, q_hi_raw)

    lo = np.clip(q_lo_raw - qhat, 0.0, None)
    hi = q_hi_raw + qhat
    hi = np.maximum(lo, hi)

    if MAX_INTERVAL_WIDTH_MONTHS is not None:
        cap = float(MAX_INTERVAL_WIDTH_MONTHS)
        hi  = np.minimum(hi, lo + cap)

    return q_lo_raw, q_hi_raw, lo, hi


def run_cqr_full(
    X_tr:  np.ndarray, t_tr:  np.ndarray, e_tr:  np.ndarray,
    X_cal: np.ndarray, t_cal: np.ndarray, e_cal: np.ndarray,
    X_te:  np.ndarray, t_te:  np.ndarray, e_te:  np.ndarray,
    alpha: float = CONFORMAL_ALPHA,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    float, float, Dict[str, Any],
]:
    """
    Full CQR pipeline: fit → calibrate → predict.

    Returns
    -------
    q_lo_raw, q_hi_raw : raw quantile predictions for test rows
    lo, hi             : conformal intervals for test rows
    qhat               : conformal correction in raw months
    coverage           : empirical coverage on test event rows
    details            : audit dictionary
    """
    n_te = len(X_te)
    nan_arr = lambda: np.full(n_te, np.nan, dtype=float)

    tr_mask  = e_tr  == 1
    cal_mask = e_cal == 1
    te_mask  = e_te  == 1
    n_tr_ev  = int(tr_mask.sum())
    n_cal_ev = int(cal_mask.sum())
    n_te_ev  = int(te_mask.sum())

    details: Dict[str, Any] = {
        "method":                   "CQR_Romano_Patterson_Candes_NeurIPS2019",
        "scale":                    "raw_months_no_log_transform",
        "alpha":                    float(alpha),
        "alpha_lo_quantile":        float(alpha / 2),
        "alpha_hi_quantile":        float(1 - alpha / 2),
        "n_train_events":           n_tr_ev,
        "n_cal_events":             n_cal_ev,
        "n_test_events":            n_te_ev,
        "coverage_claimed_for":     "event-only test rows under exchangeability",
        "status":                   "ok",
    }

    if n_tr_ev < CQR_MIN_TRAIN_EVENTS:
        details["status"] = f"skipped: {n_tr_ev} train events < {CQR_MIN_TRAIN_EVENTS}"
        return nan_arr(), nan_arr(), nan_arr(), nan_arr(), np.nan, np.nan, details
    if n_cal_ev < CQR_MIN_CAL_EVENTS:
        details["status"] = f"skipped: {n_cal_ev} cal events < {CQR_MIN_CAL_EVENTS}"
        return nan_arr(), nan_arr(), nan_arr(), nan_arr(), np.nan, np.nan, details

    X_tr_ev  = X_tr[tr_mask];   y_tr_ev  = t_tr[tr_mask]
    X_cal_ev = X_cal[cal_mask]; y_cal_ev = t_cal[cal_mask]

    try:
        qr_lo, qr_hi = fit_cqr_models(X_tr_ev, y_tr_ev, alpha)
    except Exception as exc:
        details["status"] = f"skipped: QR fit failed ({exc})"
        return nan_arr(), nan_arr(), nan_arr(), nan_arr(), np.nan, np.nan, details

    _, _, scores = compute_cqr_scores(qr_lo, qr_hi, X_cal_ev, y_cal_ev)

    details.update({
        "calibration_score_min":    float(np.min(scores)),
        "calibration_score_median": float(np.median(scores)),
        "calibration_score_p90":    float(np.quantile(scores, 0.90)),
        "calibration_score_max":    float(np.max(scores)),
        "calibration_score_mean":   float(np.mean(scores)),
    })

    qhat = conformal_quantile(scores, alpha)
    details["qhat_months"] = float(qhat)

    if not np.isfinite(qhat):
        details["status"] = f"skipped: qhat=+inf (n_cal_events={n_cal_ev})"
        return nan_arr(), nan_arr(), nan_arr(), nan_arr(), np.nan, np.nan, details

    q_lo_raw, q_hi_raw, lo, hi = apply_cqr_to_test(qr_lo, qr_hi, X_te, qhat)

    all_widths = hi - lo
    details.update({
        "mean_interval_width_months_all_test":    float(np.mean(all_widths)),
        "median_interval_width_months_all_test":  float(np.median(all_widths)),
        "p90_interval_width_months_all_test":     float(np.percentile(all_widths, 90)),
    })

    if n_te_ev > 0:
        y_te_ev  = t_te[te_mask]
        lo_ev    = lo[te_mask];   hi_ev = hi[te_mask]
        coverage = float(np.mean((y_te_ev >= lo_ev) & (y_te_ev <= hi_ev)))
        details.update({
            "coverage_test_events_only":              coverage,
            "mean_interval_width_months_test_events": float(np.mean(hi_ev - lo_ev)),
        })
    else:
        coverage = np.nan
        details["coverage_test_events_only"] = float("nan")

    return q_lo_raw, q_hi_raw, lo, hi, qhat, coverage, details


# ─────────────────────────────────────────────────────────────────────────────
# High-uncertainty flagging
# ─────────────────────────────────────────────────────────────────────────────

def flag_high_uncertainty(
    q_lo_raw: np.ndarray,
    q_hi_raw: np.ndarray,
    lo:       np.ndarray,
    hi:       np.ndarray,
    risk_scores: np.ndarray,
    uncertainty_quantile: float = UNCERTAINTY_QUANTILE,
) -> pd.DataFrame:
    """
    Classify each test patient along three uncertainty axes and produce
    a structured uncertainty DataFrame.

    Parameters
    ----------
    q_lo_raw, q_hi_raw : raw GBR quantile predictions (before conformal)
    lo, hi             : conformal interval bounds
    risk_scores        : Cox risk scores for test patients
    uncertainty_quantile : threshold quantile for flagging (default 0.75)

    Returns
    -------
    DataFrame with one row per test patient and columns:
      qr_lo_raw, qr_hi_raw, qr_width_raw,
      conformal_lo, conformal_hi, conformal_width,
      type1_model_uncertain, type2_risk_ambiguous, type3_upper_tail,
      n_uncertainty_types, high_uncertainty, very_high_uncertainty,
      uncertainty_label, uncertainty_rank
    """
    qr_width_raw    = q_hi_raw - q_lo_raw
    conformal_width = hi - lo

    q = uncertainty_quantile

    # ── Type 1: wide raw QR interval ─────────────────────────────────────
    # The quantile models themselves disagree — aleatoric uncertainty.
    t1_threshold     = float(np.nanpercentile(qr_width_raw, q * 100))
    type1_model_unc  = qr_width_raw > t1_threshold

    # ── Type 2: mid-range risk score (ambiguous Cox ranking) ──────────────
    # The IQR of risk scores is where discriminative power is lowest.
    r_q25 = float(np.nanpercentile(risk_scores, 25))
    r_q75 = float(np.nanpercentile(risk_scores, 75))
    type2_risk_amb   = (risk_scores >= r_q25) & (risk_scores <= r_q75)

    # ── Type 3: high upper tail (long predicted survival, poorly bounded) ─
    # hi > Q75 of test hi values — model predicts long but uncertain survival.
    t3_threshold     = float(np.nanpercentile(hi, q * 100))
    type3_upper_tail = hi > t3_threshold

    # ── Composite score ───────────────────────────────────────────────────
    n_types = (
        type1_model_unc.astype(int)
        + type2_risk_amb.astype(int)
        + type3_upper_tail.astype(int)
    )
    high_uncertainty      = n_types >= 1
    very_high_uncertainty = n_types >= 2

    # Human-readable label
    labels = []
    for t1, t2, t3 in zip(type1_model_unc, type2_risk_amb, type3_upper_tail):
        parts = []
        if t1: parts.append("model_uncertain")
        if t2: parts.append("risk_ambiguous")
        if t3: parts.append("upper_tail")
        labels.append(", ".join(parts) if parts else "low_uncertainty")

    # Rank: 1 = most uncertain (first sort by n_types desc, then width desc)
    sort_keys = pd.DataFrame({
        "n": n_types,
        "w": conformal_width,
    })
    ranks = sort_keys.apply(
        lambda row: row["n"] * 1e9 + row["w"], axis=1
    ).rank(method="min", ascending=False).astype(int)

    df = pd.DataFrame({
        "qr_lo_raw":            q_lo_raw,
        "qr_hi_raw":            q_hi_raw,
        "qr_width_raw":         qr_width_raw,
        "conformal_lo":         lo,
        "conformal_hi":         hi,
        "conformal_width":      conformal_width,
        "type1_model_uncertain":type1_model_unc,
        "type2_risk_ambiguous": type2_risk_amb,
        "type3_upper_tail":     type3_upper_tail,
        "n_uncertainty_types":  n_types,
        "high_uncertainty":     high_uncertainty,
        "very_high_uncertainty":very_high_uncertainty,
        "uncertainty_label":    labels,
        "uncertainty_rank":     ranks,
        # Threshold metadata for reproducibility
        "_t1_threshold_months": t1_threshold,
        "_t2_risk_q25":         r_q25,
        "_t2_risk_q75":         r_q75,
        "_t3_threshold_hi_months": t3_threshold,
        "_uncertainty_quantile": uncertainty_quantile,
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Summary reporting
# ─────────────────────────────────────────────────────────────────────────────

def build_uncertainty_summary(
    unc_df: pd.DataFrame,
    cqr_details: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a JSON-serialisable summary of the uncertainty analysis.
    """
    n_total = len(unc_df)

    def pct(mask: pd.Series) -> float:
        return float(mask.sum() / n_total * 100) if n_total > 0 else 0.0

    summary: Dict[str, Any] = {
        "n_test_patients": int(n_total),
        "conformal_details": cqr_details,
        "uncertainty_thresholds": {
            "uncertainty_quantile_used":  float(unc_df["_uncertainty_quantile"].iloc[0]),
            "type1_qr_width_threshold_months": float(unc_df["_t1_threshold_months"].iloc[0]),
            "type2_risk_score_iqr":       {
                "q25": float(unc_df["_t2_risk_q25"].iloc[0]),
                "q75": float(unc_df["_t2_risk_q75"].iloc[0]),
            },
            "type3_hi_threshold_months":  float(unc_df["_t3_threshold_hi_months"].iloc[0]),
        },
        "flags": {
            "type1_model_uncertain":  {
                "n": int(unc_df["type1_model_uncertain"].sum()),
                "pct": pct(unc_df["type1_model_uncertain"]),
                "description": "Raw QR interval width > Q75 — model has high aleatoric uncertainty",
            },
            "type2_risk_ambiguous":   {
                "n": int(unc_df["type2_risk_ambiguous"].sum()),
                "pct": pct(unc_df["type2_risk_ambiguous"]),
                "description": "Cox risk score in IQR [Q25, Q75] — borderline discriminative zone",
            },
            "type3_upper_tail":       {
                "n": int(unc_df["type3_upper_tail"].sum()),
                "pct": pct(unc_df["type3_upper_tail"]),
                "description": "Conformal hi > Q75 of test hi — long predicted survival, poorly bounded",
            },
            "high_uncertainty":       {
                "n": int(unc_df["high_uncertainty"].sum()),
                "pct": pct(unc_df["high_uncertainty"]),
                "description": "Triggered >= 1 uncertainty type",
            },
            "very_high_uncertainty":  {
                "n": int(unc_df["very_high_uncertainty"].sum()),
                "pct": pct(unc_df["very_high_uncertainty"]),
                "description": "Triggered >= 2 uncertainty types — highest clinical attention needed",
            },
        },
        "interval_statistics": {
            "all_patients": {
                "mean_conformal_width_months":   float(unc_df["conformal_width"].mean()),
                "median_conformal_width_months": float(unc_df["conformal_width"].median()),
                "p90_conformal_width_months":    float(unc_df["conformal_width"].quantile(0.90)),
                "mean_qr_raw_width_months":      float(unc_df["qr_width_raw"].mean()),
                "median_qr_raw_width_months":    float(unc_df["qr_width_raw"].median()),
            },
            "high_uncertainty_only": (
                {
                    "mean_conformal_width_months":   float(
                        unc_df.loc[unc_df["high_uncertainty"], "conformal_width"].mean()
                    ),
                    "median_conformal_width_months": float(
                        unc_df.loc[unc_df["high_uncertainty"], "conformal_width"].median()
                    ),
                }
                if unc_df["high_uncertainty"].any() else {}
            ),
            "low_uncertainty_only": (
                {
                    "mean_conformal_width_months":   float(
                        unc_df.loc[~unc_df["high_uncertainty"], "conformal_width"].mean()
                    ),
                    "median_conformal_width_months": float(
                        unc_df.loc[~unc_df["high_uncertainty"], "conformal_width"].median()
                    ),
                }
                if (~unc_df["high_uncertainty"]).any() else {}
            ),
        },
        "type_co_occurrence": {
            "t1_only":    int(
                (unc_df["type1_model_uncertain"]
                & ~unc_df["type2_risk_ambiguous"]
                & ~unc_df["type3_upper_tail"]).sum()
            ),
            "t2_only":    int(
                (unc_df["type2_risk_ambiguous"]
                & ~unc_df["type1_model_uncertain"]
                & ~unc_df["type3_upper_tail"]).sum()
            ),
            "t3_only":    int(
                (unc_df["type3_upper_tail"]
                & ~unc_df["type1_model_uncertain"]
                & ~unc_df["type2_risk_ambiguous"]).sum()
            ),
            "t1_and_t2":  int((unc_df["type1_model_uncertain"] & unc_df["type2_risk_ambiguous"]).sum()),
            "t1_and_t3":  int((unc_df["type1_model_uncertain"] & unc_df["type3_upper_tail"]).sum()),
            "t2_and_t3":  int((unc_df["type2_risk_ambiguous"]  & unc_df["type3_upper_tail"]).sum()),
            "all_three":  int(
                (unc_df["type1_model_uncertain"]
                 & unc_df["type2_risk_ambiguous"]
                 & unc_df["type3_upper_tail"]).sum()
            ),
            "none":       int((unc_df["n_uncertainty_types"] == 0).sum()),
        },
    }
    return summary


def build_markdown_report(
    summary: Dict[str, Any],
    top_uncertain: pd.DataFrame,
) -> str:
    """
    Build a human-readable Markdown report of the uncertainty analysis.
    """
    cqr = summary["conformal_details"]
    thr = summary["uncertainty_thresholds"]
    fl  = summary["flags"]
    istat = summary["interval_statistics"]
    cooc  = summary["type_co_occurrence"]

    n = summary["n_test_patients"]
    lines: List[str] = [
        "# Survival Conformal Uncertainty Report",
        "",
        "## Conformal Method",
        "CQR (Romano, Patterson & Candès, NeurIPS 2019) on the **raw month scale**.",
        "No log transform — intervals are symmetric in months, not exponentially skewed.",
        "",
        f"- alpha             : {cqr['alpha']} (target coverage {int((1-cqr['alpha'])*100)}%)",
        f"- qhat              : {cqr.get('qhat_months', 'N/A'):.2f} months"
          if isinstance(cqr.get('qhat_months'), float) and np.isfinite(cqr.get('qhat_months', np.nan))
          else f"- qhat              : N/A",
        f"- status            : {cqr['status']}",
        f"- coverage (events) : {cqr.get('coverage_test_events_only', 'N/A')}",
        "",
        "## Interval Width Summary",
        f"| Group | Mean width (months) | Median width (months) |",
        f"|---|---|---|",
        f"| All test patients | {istat['all_patients']['mean_conformal_width_months']:.1f} | "
          f"{istat['all_patients']['median_conformal_width_months']:.1f} |",
    ]
    if istat.get("high_uncertainty_only"):
        lines.append(
            f"| High-uncertainty | {istat['high_uncertainty_only']['mean_conformal_width_months']:.1f} | "
            f"{istat['high_uncertainty_only']['median_conformal_width_months']:.1f} |"
        )
    if istat.get("low_uncertainty_only"):
        lines.append(
            f"| Low-uncertainty  | {istat['low_uncertainty_only']['mean_conformal_width_months']:.1f} | "
            f"{istat['low_uncertainty_only']['median_conformal_width_months']:.1f} |"
        )

    lines += [
        "",
        "## Uncertainty Flags",
        f"Total test patients: **{n}**",
        "",
        "| Type | N | % | Description |",
        "|---|---|---|---|",
    ]
    for key in ["type1_model_uncertain", "type2_risk_ambiguous", "type3_upper_tail",
                "high_uncertainty", "very_high_uncertainty"]:
        f = fl[key]
        lines.append(f"| {key} | {f['n']} | {f['pct']:.1f}% | {f['description']} |")

    lines += [
        "",
        "## Uncertainty Thresholds",
        f"- Uncertainty quantile: {thr['uncertainty_quantile_used']} (top "
          f"{int((1-thr['uncertainty_quantile_used'])*100)}% flagged per axis)",
        f"- Type 1 threshold (QR raw width): > {thr['type1_qr_width_threshold_months']:.1f} months",
        f"- Type 2 threshold (risk IQR): [{thr['type2_risk_score_iqr']['q25']:.3f}, "
          f"{thr['type2_risk_score_iqr']['q75']:.3f}]",
        f"- Type 3 threshold (conformal hi): > {thr['type3_hi_threshold_months']:.1f} months",
        "",
        "## Type Co-occurrence",
        f"| Pattern | N |",
        f"|---|---|",
        f"| Type 1 only (model_uncertain) | {cooc['t1_only']} |",
        f"| Type 2 only (risk_ambiguous)  | {cooc['t2_only']} |",
        f"| Type 3 only (upper_tail)      | {cooc['t3_only']} |",
        f"| Type 1 + 2                    | {cooc['t1_and_t2']} |",
        f"| Type 1 + 3                    | {cooc['t1_and_t3']} |",
        f"| Type 2 + 3                    | {cooc['t2_and_t3']} |",
        f"| All three                     | {cooc['all_three']} |",
        f"| None (low uncertainty)        | {cooc['none']} |",
        "",
        "## Top 20 Most Uncertain Patients",
        top_uncertain[[
            "PATIENT_ID", "OS_MONTHS", "OS_EVENT", "risk_score",
            "conformal_lo", "conformal_hi", "conformal_width",
            "n_uncertainty_types", "uncertainty_label",
        ]].to_markdown(index=False),
        "",
        "## Output Files",
        "- `conformal_uncertainty_predictions.csv` — all test patients with uncertainty columns",
        "- `high_uncertainty_patients.csv`          — only high_uncertainty patients, sorted by rank",
        "- `very_high_uncertainty_patients.csv`     — very_high_uncertainty patients only",
        "- `uncertainty_summary.json`               — full numeric summary",
        "- `uncertainty_report.md`                  — this file",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Feature reconstruction from saved model artefact
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_feature_matrices(
    artifact: Dict[str, Any],
    df: pd.DataFrame,
    idx_train: np.ndarray,
    idx_cal:   np.ndarray,
    idx_test:  np.ndarray,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:
    """
    Reconstruct X matrices and target arrays for train/cal/test splits
    using the preprocessors stored in the model artifact.
    """
    clin_pre  = artifact["clin_pre"]
    expr_pipe = artifact["expr_pipe"]
    scaler    = artifact["scaler"]

    clinical_cols = artifact["clinical_cols_after_collinearity"]
    expr_cols     = artifact["expr_cols_after_collinearity"]

    def _transform(df_split: pd.DataFrame) -> np.ndarray:
        blocks = []
        if clinical_cols:
            blocks.append(
                np.asarray(clin_pre.transform(df_split[clinical_cols].copy()), dtype=float)
            )
        if expr_cols:
            blocks.append(
                np.asarray(expr_pipe.transform(df_split[expr_cols].copy()), dtype=float)
            )
        if not blocks:
            raise ValueError("No feature blocks to transform.")
        return scaler.transform(np.hstack(blocks))

    tr  = df.iloc[idx_train]
    cal = df.iloc[idx_cal]
    te  = df.iloc[idx_test]

    X_tr  = _transform(tr)
    X_cal = _transform(cal)
    X_te  = _transform(te)

    t_tr  = tr["OS_MONTHS"].to_numpy(float);   e_tr  = tr["OS_EVENT"].to_numpy(int)
    t_cal = cal["OS_MONTHS"].to_numpy(float);  e_cal = cal["OS_EVENT"].to_numpy(int)
    t_te  = te["OS_MONTHS"].to_numpy(float);   e_te  = te["OS_EVENT"].to_numpy(int)

    return X_tr, X_cal, X_te, t_tr, t_cal, t_te, e_tr, e_cal, e_te


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model artifact ────────────────────────────────────────────
    print(f"Loading model artifact: {MODEL_PKL}")
    with open(MODEL_PKL, "rb") as f:
        artifact = pickle.load(f)

    cox_model = artifact["cox_model"]

    # ── 2. Load data and reconstruct splits ───────────────────────────────
    print(f"Loading data: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    df["OS_MONTHS"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    df["OS_EVENT"]  = pd.to_numeric(df["OS_EVENT"],  errors="coerce")
    df = df.dropna(subset=["OS_MONTHS", "OS_EVENT"]).copy()
    df = df[df["OS_MONTHS"] > 0].copy()
    df["OS_EVENT"] = (df["OS_EVENT"] > 0).astype(int)
    df = df.reset_index(drop=True)

    if "PATIENT_ID" not in df.columns:
        df["PATIENT_ID"] = [f"ROW_{i:07d}" for i in range(len(df))]
    df["PATIENT_ID"] = df["PATIENT_ID"].astype(str)

    # Reconstruct the same 60/20/20 split deterministically
    from sklearn.model_selection import train_test_split
    idx    = np.arange(len(df), dtype=int)
    y      = df["OS_EVENT"].to_numpy(int)
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.4, random_state=42, stratify=y)
    idx_cal, idx_te = train_test_split(idx_tmp, test_size=0.5, random_state=42, stratify=y[idx_tmp])

    print(f"Split: train={len(idx_tr)} | cal={len(idx_cal)} | test={len(idx_te)}")

    (X_tr, X_cal, X_te,
     t_tr, t_cal, t_te,
     e_tr, e_cal, e_te) = reconstruct_feature_matrices(
        artifact, df, idx_tr, idx_cal, idx_te
    )

    # ── 3. Cox risk scores ────────────────────────────────────────────────
    r_tr  = cox_model.predict_risk(X_tr)
    r_cal = cox_model.predict_risk(X_cal)
    r_te  = cox_model.predict_risk(X_te)

    # ── 4. CQR intervals on raw month scale ───────────────────────────────
    print("Fitting CQR quantile models and computing conformal intervals …")
    q_lo_raw, q_hi_raw, lo, hi, qhat, coverage, cqr_details = run_cqr_full(
        X_tr, t_tr, e_tr,
        X_cal, t_cal, e_cal,
        X_te,  t_te,  e_te,
        alpha=CONFORMAL_ALPHA,
    )
    print(f"CQR: status={cqr_details['status']}")
    if np.isfinite(qhat):
        print(f"  qhat = {qhat:.2f} months")
    if np.isfinite(coverage):
        print(f"  Coverage (events) = {coverage:.3f}")

    # ── 5. Uncertainty flagging ───────────────────────────────────────────
    print("Flagging high-uncertainty patients …")
    unc_df = flag_high_uncertainty(
        q_lo_raw=q_lo_raw,
        q_hi_raw=q_hi_raw,
        lo=lo,
        hi=hi,
        risk_scores=r_te,
        uncertainty_quantile=UNCERTAINTY_QUANTILE,
    )

    # ── 6. Assemble full test predictions ─────────────────────────────────
    te_df = df.iloc[idx_te].copy().reset_index(drop=True)
    te_df["risk_score"] = r_te
    te_df = pd.concat([te_df.reset_index(drop=True), unc_df.reset_index(drop=True)], axis=1)

    # Drop internal threshold columns from main CSV but keep in summary
    threshold_cols = [c for c in te_df.columns if c.startswith("_")]
    te_export = te_df.drop(columns=threshold_cols)

    out_full = OUT_DIR / "conformal_uncertainty_predictions.csv"
    te_export.to_csv(out_full, index=False)
    print(f"Saved full predictions: {out_full}")

    # ── 7. High-uncertainty subsets ───────────────────────────────────────
    high_unc = te_export[te_export["high_uncertainty"]].sort_values(
        ["n_uncertainty_types", "conformal_width"], ascending=[False, False]
    )
    very_high_unc = te_export[te_export["very_high_uncertainty"]].sort_values(
        ["n_uncertainty_types", "conformal_width"], ascending=[False, False]
    )

    out_high = OUT_DIR / "high_uncertainty_patients.csv"
    out_very  = OUT_DIR / "very_high_uncertainty_patients.csv"
    high_unc.to_csv(out_high,  index=False)
    very_high_unc.to_csv(out_very, index=False)
    print(f"High-uncertainty patients:      {len(high_unc)} → {out_high}")
    print(f"Very-high-uncertainty patients: {len(very_high_unc)} → {out_very}")

    # ── 8. Summary JSON ───────────────────────────────────────────────────
    summary = build_uncertainty_summary(unc_df, cqr_details)
    out_json = OUT_DIR / "uncertainty_summary.json"

    # JSON-serialise numpy types
    def _jsonify(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        if isinstance(obj, dict):           return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):           return [_jsonify(i) for i in obj]
        return obj

    out_json.write_text(
        json.dumps(_jsonify(summary), indent=2), encoding="utf-8"
    )
    print(f"Summary JSON: {out_json}")

    # ── 9. Markdown report ────────────────────────────────────────────────
    top20 = te_export.nsmallest(20, "uncertainty_rank")
    md    = build_markdown_report(summary, top20)
    out_md = OUT_DIR / "uncertainty_report.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"Markdown report: {out_md}")

    # ── 10. Console summary ───────────────────────────────────────────────
    fl = summary["flags"]
    print("\n" + "=" * 60)
    print("UNCERTAINTY SUMMARY")
    print("=" * 60)
    print(f"  Total test patients:        {summary['n_test_patients']}")
    print(f"  High uncertainty (≥1 type): {fl['high_uncertainty']['n']} "
          f"({fl['high_uncertainty']['pct']:.1f}%)")
    print(f"  Very high (≥2 types):       {fl['very_high_uncertainty']['n']} "
          f"({fl['very_high_uncertainty']['pct']:.1f}%)")
    print(f"  Type 1 (model uncertain):   {fl['type1_model_uncertain']['n']} "
          f"({fl['type1_model_uncertain']['pct']:.1f}%)")
    print(f"  Type 2 (risk ambiguous):    {fl['type2_risk_ambiguous']['n']} "
          f"({fl['type2_risk_ambiguous']['pct']:.1f}%)")
    print(f"  Type 3 (upper tail):        {fl['type3_upper_tail']['n']} "
          f"({fl['type3_upper_tail']['pct']:.1f}%)")
    istat = summary["interval_statistics"]
    print(f"\n  Interval width (all test):")
    print(f"    Mean   = {istat['all_patients']['mean_conformal_width_months']:.1f} months")
    print(f"    Median = {istat['all_patients']['median_conformal_width_months']:.1f} months")
    print(f"    P90    = {istat['all_patients']['p90_conformal_width_months']:.1f} months")
    if istat.get("high_uncertainty_only"):
        print(f"\n  High-uncertainty patients only:")
        print(f"    Mean   = {istat['high_uncertainty_only']['mean_conformal_width_months']:.1f} months")
        print(f"    Median = {istat['high_uncertainty_only']['median_conformal_width_months']:.1f} months")
    print("=" * 60)
    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
