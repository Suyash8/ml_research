"""
===============================================================================
MULTI-OMIC INFERENCE PIPELINE
===============================================================================
Encapsulates real-time inference, isotonic calibration, Monte Carlo stochastic
simulations, and gene-level explainability for the Cox Elastic-Net model.
"""

import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.data.loader import prepare_dataframe
from src.features.preprocessing import transform_features
from src.inference.explainability import ExplainabilityModule
from src.metrics.explainability import build_feature_names
from src.training.monte_carlo import fit_breslow_baseline_hazard, simulate_cox_survival_times
from src.utils.config import HORIZONS_MONTHS, MC_N_SIMS, MC_RANDOM_STATE


class MultiOmicInferencePipeline:
    """Unified Inference Pipeline for Multi-Omic Cox Elastic-Net Model."""

    def __init__(self, artifact_path: Union[str, Path]):
        self.artifact_path = Path(artifact_path)
        self.artifact: Dict[str, Any] = {}
        self.is_loaded = False
        
        self.clinical_cols: List[str] = []
        self.expr_cols: List[str] = []
        self.clin_pre = None
        self.expr_pipe = None
        self.scaler = None
        self.cox_model = None
        self.calibrators: Dict[Any, Any] = {}

        self.baseline_times: Optional[np.ndarray] = None
        self.baseline_cumhaz: Optional[np.ndarray] = None
        self.max_followup_months: float = 60.0

        self.load_artifact()

    def load_artifact(self) -> None:
        """Loads frozen pickle artifact and initializes model parameters."""
        if not self.artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {self.artifact_path}")

        # Ensure CoxElasticNet class is registered in __main__ module for pickle loading
        from src.models.cox_enet import CoxElasticNet
        main_module = sys.modules.get("__main__")
        if main_module is not None and not hasattr(main_module, "CoxElasticNet"):
            setattr(main_module, "CoxElasticNet", CoxElasticNet)

        with open(self.artifact_path, "rb") as f:
            self.artifact = pickle.load(f)

        self.clinical_cols = list(self.artifact.get("clinical_cols_after_collinearity") or [])
        self.expr_cols = list(self.artifact.get("expr_cols_after_collinearity") or [])
        self.clin_pre = self.artifact.get("clin_pre")
        self.expr_pipe = self.artifact.get("expr_pipe")
        self.scaler = self.artifact.get("scaler")
        self.cox_model = self.artifact.get("cox_model")
        self.calibrators = self.artifact.get("horizon_calibrators", {})
        self.is_loaded = True

        self._fit_baseline_hazard()

    def _fit_baseline_hazard(self) -> None:
        """Fits Breslow baseline hazard from preprocessed training set for Monte Carlo simulations."""
        root_dir = self.artifact_path.resolve().parent.parent
        clean_parquet_path = root_dir / "data" / "preprocessed_cleaned" / "patient_multiomic_cleaned.parquet"
        
        if clean_parquet_path.exists():
            df_full = prepare_dataframe(clean_parquet_path)
            X_full = transform_features(
                df_full, self.clinical_cols, self.expr_cols,
                self.clin_pre, self.expr_pipe, self.scaler
            )
            risk_scores_full = np.asarray(self.cox_model.predict_risk(X_full), dtype=float)
            times = df_full["OS_MONTHS"].to_numpy(dtype=float)
            events = df_full["OS_EVENT"].to_numpy(dtype=int)

            b_times, b_cumhaz, meta = fit_breslow_baseline_hazard(times, events, risk_scores_full)
            self.baseline_times = b_times
            self.baseline_cumhaz = b_cumhaz
            self.max_followup_months = meta.get("max_observed_followup_months", float(np.max(times)))
        else:
            # Fallback baseline hazard profile
            self.baseline_times = np.array([12.0, 24.0, 36.0, 60.0])
            self.baseline_cumhaz = np.array([0.1, 0.3, 0.5, 0.8])
            self.max_followup_months = 60.0

    def transform_input(self, df_input: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Transforms raw input DataFrame into scaled feature matrix."""
        clin_names, expr_names, feature_names = build_feature_names(self.clin_pre, self.clinical_cols, self.expr_pipe)
        X = transform_features(
            df_input, self.clinical_cols, self.expr_cols,
            self.clin_pre, self.expr_pipe, self.scaler
        )
        return X, feature_names

    def predict_risk(self, df_input: pd.DataFrame) -> np.ndarray:
        """Computes continuous log-hazard risk scores (eta)."""
        X, _ = self.transform_input(df_input)
        return np.asarray(self.cox_model.predict_risk(X), dtype=float)

    def predict_calibrated_survival(
        self, df_input: pd.DataFrame, horizons: List[float] = HORIZONS_MONTHS
    ) -> pd.DataFrame:
        """Computes isotonic-calibrated survival probabilities at specified month horizons."""
        risk_scores = self.predict_risk(df_input)
        probs_dict = {}
        for h in horizons:
            cal_info = self.calibrators.get(float(h))
            if isinstance(cal_info, dict) and "isotonic" in cal_info and cal_info["isotonic"] is not None:
                iso_model = cal_info["isotonic"]
                event_probs = iso_model.predict(risk_scores)
                surv_probs = np.clip(1.0 - event_probs, 0.0, 1.0)
                probs_dict[f"prob_survive_{int(h)}m"] = surv_probs
            else:
                probs_dict[f"prob_survive_{int(h)}m"] = np.full(len(risk_scores), np.nan)
        return pd.DataFrame(probs_dict)

    def simulate_survival(
        self, df_input: pd.DataFrame, n_sims: int = MC_N_SIMS, random_state: int = MC_RANDOM_STATE
    ) -> pd.DataFrame:
        """Runs Monte Carlo inverse transform sampling to get P10, P50, P90 & RMST."""
        risk_scores = self.predict_risk(df_input)
        sim_times = simulate_cox_survival_times(
            risk_scores,
            baseline_times=self.baseline_times,
            baseline_cumhaz=self.baseline_cumhaz,
            max_followup_months=self.max_followup_months,
            n_sims=n_sims,
            random_state=random_state,
        )
        
        rows = []
        for i in range(sim_times.shape[0]):
            s = sim_times[i]
            rows.append({
                "mc_p10_months": float(np.quantile(s, 0.10)),
                "mc_p50_median_months": float(np.quantile(s, 0.50)),
                "mc_p90_months": float(np.quantile(s, 0.90)),
                "mc_rmst_months": float(np.mean(np.minimum(s, 60.0))),
            })
        return pd.DataFrame(rows)

    def explain(self, df_input: pd.DataFrame, top_n_drivers: int = 5) -> List[Dict[str, Any]]:
        """Generates patient-level explainability waterfalled into clinical & gene drivers."""
        X, feature_names = self.transform_input(df_input)
        coefs = np.asarray(self.cox_model.coef_, dtype=float)

        pca = None
        if self.expr_pipe is not None and hasattr(self.expr_pipe, "named_steps"):
            pca = self.expr_pipe.named_steps.get("pca")

        pca_loadings = getattr(pca, "components_", None) if pca is not None else None

        explain_mod = ExplainabilityModule(
            clinical_feature_names=[name for name in feature_names if not name.startswith("EXPR_PC")],
            expr_cols=self.expr_cols,
            coefs=coefs,
            pca_loadings=pca_loadings
        )

        patient_ids = (
            df_input["PATIENT_ID"].astype(str).tolist()
            if "PATIENT_ID" in df_input.columns
            else [f"PATIENT_{i+1:03d}" for i in range(len(df_input))]
        )

        n_clin = len(explain_mod.clinical_feature_names)
        results = []
        for i, pid in enumerate(patient_ids):
            X_clin_patient = X[i, :n_clin]
            
            # Extract raw gene expressions for patient if present
            raw_expr = np.zeros(len(self.expr_cols))
            for g_idx, g_col in enumerate(self.expr_cols):
                if g_col in df_input.columns:
                    raw_expr[g_idx] = float(df_input[g_col].iloc[i])

            exp = explain_mod.explain_patient(pid, X_clin_patient, raw_expr, top_n=top_n_drivers)
            results.append(exp)

        return results

    def predict(
        self, df_input: pd.DataFrame, horizons: List[float] = HORIZONS_MONTHS, n_sims: int = MC_N_SIMS
    ) -> pd.DataFrame:
        """
        Unified Inference Execution: Returns complete prediction DataFrame with
        Risk Scores, Calibrated Probabilities, Monte Carlo Bounds, and Top Explainability Drivers.
        """
        patient_ids = (
            df_input["PATIENT_ID"].astype(str).tolist()
            if "PATIENT_ID" in df_input.columns
            else [f"PATIENT_{i+1:03d}" for i in range(len(df_input))]
        )

        # 1. Risk Score
        risk_scores = self.predict_risk(df_input)

        # 2. Calibrated Survival Probabilities
        df_cal = self.predict_calibrated_survival(df_input, horizons=horizons)

        # 3. Monte Carlo Bounds
        df_mc = self.simulate_survival(df_input, n_sims=n_sims)

        # 4. Explainability Summaries
        explanations = self.explain(df_input, top_n_drivers=3)
        top_risk = [", ".join(d["feature"] for d in exp["top_risk_drivers"]) for exp in explanations]
        top_protective = [", ".join(d["feature"] for d in exp["top_protective_drivers"]) for exp in explanations]

        # Combine into single DataFrame
        df_out = pd.DataFrame({
            "PATIENT_ID": patient_ids,
            "risk_score_eta": risk_scores,
        })

        for col in df_cal.columns:
            df_out[col] = df_cal[col]

        for col in df_mc.columns:
            df_out[col] = df_mc[col]

        df_out["top_risk_drivers"] = top_risk
        df_out["top_protective_drivers"] = top_protective

        return df_out
