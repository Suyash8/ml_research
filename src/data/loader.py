from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ml_research.src.utils.config import OUTLIER_COLUMNS, OUTLIER_IQR_MULTIPLIER, RANDOM_STATE
from ml_research.src.utils.io import safe_float
from pathlib import Path

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
