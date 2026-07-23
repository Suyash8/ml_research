from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from src.utils.config import HORIZONS_MONTHS, MC_N_SIMS, MC_RANDOM_STATE, MC_RMST_HORIZON_MONTHS

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
