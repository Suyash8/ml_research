from typing import Any, Dict, Sequence, Tuple
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from ml_research.src.utils.config import HORIZONS_MONTHS

def _horizon_labels(time_arr: np.ndarray, event_arr: np.ndarray, horizon: float) -> Tuple[np.ndarray, np.ndarray]:
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
