from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from ml_research.src.models.calibration import _horizon_labels
from ml_research.src.utils.config import HORIZONS_MONTHS

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
