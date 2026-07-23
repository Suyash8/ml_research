from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src.utils.config import ALPHA_GRID, CV_FOLDS, L1_GRID, MAXITER, RANDOM_STATE
from src.features.preprocessing import fit_transform_features
from src.models.cox_enet import CoxElasticNet
from src.metrics.survival import concordance_index_censored

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
