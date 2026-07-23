from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils.config import EXPR_PCA_COMPONENTS, RANDOM_STATE

def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def _fit_clinical_block(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    clinical_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, Optional[ColumnTransformer], List[str], List[str]]:
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
    X_tr = np.asarray(pre.fit_transform(c_tr), dtype=float)
    X_va = np.asarray(pre.transform(c_va), dtype=float)
    return X_tr, X_va, pre, num_cols, cat_cols

def _fit_expression_block(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    expr_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, Optional[Pipeline], int]:
    empty = lambda n: np.empty((n, 0), dtype=float)

    if not expr_cols:
        return empty(len(df_tr)), empty(len(df_va)), None, 0

    e_tr = df_tr[expr_cols].copy()
    e_va = df_va[expr_cols].copy()
    n_comp = min(EXPR_PCA_COMPONENTS, int(e_tr.shape[0]), int(e_tr.shape[1]))
    if n_comp < 1:
        raise ValueError("Cannot configure PCA with fewer than 1 component.")

    pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("sc", StandardScaler()),
            ("pca", PCA(n_components=n_comp, random_state=RANDOM_STATE)),
        ]
    )
    X_tr = np.asarray(pipe.fit_transform(e_tr), dtype=float)
    X_va = np.asarray(pipe.transform(e_va), dtype=float)
    return X_tr, X_va, pipe, int(n_comp)

def fit_transform_features(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    clinical_cols: List[str],
    expr_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, Optional[ColumnTransformer], Optional[Pipeline], StandardScaler, Dict[str, Any]]:
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
    checks: Dict[str, Any] = {
        "clinical_num_cols": num_cols,
        "clinical_cat_cols": cat_cols,
        "clinical_output_dim": int(Xc_tr.shape[1]),
        "expr_input_dim": int(len(expr_cols)),
        "expr_pca_components": int(n_comp),
        "expr_output_dim": int(Xe_tr.shape[1]),
        "final_dim": int(X_tr.shape[1]),
        "final_train_mean_abs_max": float(np.max(np.abs(np.nanmean(X_tr, axis=0)))),
        "final_train_std_min": float(np.min(stds)),
        "final_train_std_max": float(np.max(stds)),
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
    X = scaler.transform(np.hstack(blocks))
    if not np.isfinite(X).all():
        raise ValueError("Non-finite values after transform.")
    return X
