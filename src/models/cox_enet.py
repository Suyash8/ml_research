from typing import Tuple
import numpy as np
from scipy.optimize import minimize
from ml_research.src.utils.config import MAXITER, SMOOTH_L1_EPS

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
        order = np.argsort(time, kind="mergesort")
        Xo = np.asarray(X[order], dtype=float)
        to = np.asarray(time[order], dtype=float)
        eo = np.asarray(event[order], dtype=int)

        eta = np.clip(Xo @ beta, -40, 40)
        exp_eta = np.exp(eta)

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
