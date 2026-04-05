import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
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

from train_cox_enet_conformal import CoxElasticNet, concordance_index_censored, quantile_conformal


RANDOM_STATE = 42
EXPR_PCA_COMPONENTS = 50
MAXITER = 400
CV_FOLDS = 5
CONFORMAL_ALPHA = 0.10
HORIZONS_MONTHS = [12, 24, 36, 60]

ALPHA_GRID = [0.1, 0.3, 0.8, 1.5, 3.0]
L1_GRID = [0.0, 0.1, 0.3, 0.5, 0.7]

OUTLIER_IQR_MULTIPLIER = 3.0
OUTLIER_COLUMNS = ('OS_MONTHS', 'AGE')

BASE = Path('/home/illionar/Projects/ml_research')
INPUT_PATH = BASE / 'data' / 'preprocessed_cleaned' / 'patient_multiomic_cleaned.parquet'
OUT_DIR = BASE / 'data' / 'model_outputs' / 'cox_enet_conformal_tuned_td_outlier'


def make_one_hot_encoder() -> OneHotEncoder:
    """Create a dense one-hot encoder compatible with older/newer scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)


def prepare_dataframe() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f'Input parquet not found: {INPUT_PATH}')

    df = pd.read_parquet(INPUT_PATH)
    required_cols = {'OS_MONTHS', 'OS_EVENT'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')

    df = df.copy()
    df['OS_MONTHS'] = pd.to_numeric(df['OS_MONTHS'], errors='coerce')
    df['OS_EVENT'] = pd.to_numeric(df['OS_EVENT'], errors='coerce')
    df = df.dropna(subset=['OS_MONTHS', 'OS_EVENT']).copy()
    df = df[df['OS_MONTHS'] > 0].copy()

    # Any positive indicator is treated as an observed event.
    df['OS_EVENT'] = (df['OS_EVENT'] > 0).astype(int)

    if 'PATIENT_ID' not in df.columns:
        df['PATIENT_ID'] = [f'ROW_{i:07d}' for i in range(len(df))]
    df['PATIENT_ID'] = df['PATIENT_ID'].astype(str)

    n_rows = int(len(df))
    n_events = int(df['OS_EVENT'].sum())
    n_censored = int(n_rows - n_events)
    if n_rows < 20:
        raise ValueError(f'Not enough rows after cleaning: {n_rows}. Need at least 20.')
    if n_events == 0 or n_censored == 0:
        raise ValueError('Dataset has only one class after cleaning; need both events and censored rows.')

    return df


def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: Tuple[str, ...] = OUTLIER_COLUMNS,
    iqr_multiplier: float = OUTLIER_IQR_MULTIPLIER,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Remove extreme outliers using an IQR rule and return a full audit report."""
    work = df.copy()
    report: Dict[str, Any] = {
        'initial_rows': int(len(work)),
        'iqr_multiplier': float(iqr_multiplier),
        'rules': [],
    }

    for col in columns:
        if col not in work.columns:
            report['rules'].append({
                'column': col,
                'skipped': True,
                'reason': 'column_not_found',
                'removed_rows': 0,
            })
            continue

        s = pd.to_numeric(work[col], errors='coerce')
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = float(q3 - q1)

        if not np.isfinite(iqr) or iqr <= 0:
            report['rules'].append({
                'column': col,
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'skipped': True,
                'reason': 'non_positive_or_non_finite_iqr',
                'removed_rows': 0,
            })
            continue

        lo = float(q1 - iqr_multiplier * iqr)
        hi = float(q3 + iqr_multiplier * iqr)
        keep = s.isna() | ((s >= lo) & (s <= hi))
        removed = int((~keep).sum())

        report['rules'].append({
            'column': col,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'lower_bound': lo,
            'upper_bound': hi,
            'removed_rows': removed,
            'skipped': False,
        })

        work = work.loc[keep].copy()
        if len(work) == 0:
            raise ValueError(f'Outlier filtering removed all rows while processing column {col}.')

    report['final_rows'] = int(len(work))
    report['total_removed'] = int(report['initial_rows'] - report['final_rows'])
    return work, report


def split_three_way(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Dict[str, Any]]]:
    idx = np.arange(len(df), dtype=int)
    y = df['OS_EVENT'].to_numpy(dtype=int)

    if len(np.unique(y)) < 2:
        raise ValueError('Cannot split: OS_EVENT has only one class.')

    try:
        idx_train, idx_tmp = train_test_split(
            idx,
            test_size=0.4,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        y_tmp = y[idx_tmp]
        idx_cal, idx_test = train_test_split(
            idx_tmp,
            test_size=0.5,
            random_state=RANDOM_STATE,
            stratify=y_tmp,
        )
    except ValueError as exc:
        raise ValueError(
            'Unable to create stratified train/calibration/test splits. '
            'Ensure both classes have enough samples after filtering.'
        ) from exc

    split_report: Dict[str, Dict[str, Any]] = {}
    for name, split_idx in (('train', idx_train), ('calibration', idx_cal), ('test', idx_test)):
        n_rows = int(len(split_idx))
        n_events = int(y[split_idx].sum())
        n_censored = int(n_rows - n_events)
        split_report[name] = {
            'rows': n_rows,
            'events': n_events,
            'censored': n_censored,
            'event_rate': float(n_events / n_rows) if n_rows else np.nan,
        }
        if n_events == 0 or n_censored == 0:
            raise ValueError(f'{name} split contains only one class; cannot train/evaluate robustly.')

    return idx_train, idx_cal, idx_test, split_report


def get_feature_sets(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    clinical_candidates = ['AGE', 'SEX', 'RACE', 'ETHNICITY', 'CANCER_TYPE', 'AGE_GROUP']
    clinical_cols = [c for c in clinical_candidates if c in df.columns]
    expr_cols = sorted(c for c in df.columns if c.startswith('EXPR_'))

    if not clinical_cols and not expr_cols:
        raise ValueError('No usable clinical or expression feature columns found.')

    return clinical_cols, expr_cols


def _fit_clinical_block(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    clinical_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, Optional[ColumnTransformer], List[str], List[str]]:
    if not clinical_cols:
        return (
            np.empty((len(df_tr), 0), dtype=float),
            np.empty((len(df_va), 0), dtype=float),
            None,
            [],
            [],
        )

    c_tr = df_tr[clinical_cols].copy()
    c_va = df_va[clinical_cols].copy()

    num_cols = c_tr.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    cat_cols = [c for c in c_tr.columns if c not in num_cols]

    transformers = []
    if num_cols:
        transformers.append((
            'num',
            Pipeline([
                ('imp', SimpleImputer(strategy='median')),
                ('sc', StandardScaler()),
            ]),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            'cat',
            Pipeline([
                ('imp', SimpleImputer(strategy='most_frequent')),
                ('oh', make_one_hot_encoder()),
            ]),
            cat_cols,
        ))

    if not transformers:
        return (
            np.empty((len(df_tr), 0), dtype=float),
            np.empty((len(df_va), 0), dtype=float),
            None,
            [],
            [],
        )

    clinical_pre = ColumnTransformer(transformers=transformers, remainder='drop')
    Xc_tr = np.asarray(clinical_pre.fit_transform(c_tr), dtype=float)
    Xc_va = np.asarray(clinical_pre.transform(c_va), dtype=float)
    return Xc_tr, Xc_va, clinical_pre, num_cols, cat_cols


def _fit_expression_block(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    expr_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, Optional[Pipeline], int]:
    if not expr_cols:
        return (
            np.empty((len(df_tr), 0), dtype=float),
            np.empty((len(df_va), 0), dtype=float),
            None,
            0,
        )

    e_tr = df_tr[expr_cols].copy()
    e_va = df_va[expr_cols].copy()

    n_components = min(EXPR_PCA_COMPONENTS, int(e_tr.shape[0]), int(e_tr.shape[1]))
    if n_components < 1:
        raise ValueError('Expression PCA cannot be configured with less than one component.')

    expr_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='constant', fill_value=0.0)),
        ('sc', StandardScaler()),
        ('pca', PCA(n_components=n_components, random_state=RANDOM_STATE)),
    ])

    Xe_tr = np.asarray(expr_pipe.fit_transform(e_tr), dtype=float)
    Xe_va = np.asarray(expr_pipe.transform(e_va), dtype=float)
    return Xe_tr, Xe_va, expr_pipe, int(n_components)


def fit_transform_features(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    clinical_cols: List[str],
    expr_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, Optional[ColumnTransformer], Optional[Pipeline], StandardScaler, Dict[str, Any]]:
    Xc_tr, Xc_va, clinical_pre, num_cols, cat_cols = _fit_clinical_block(df_tr, df_va, clinical_cols)
    Xe_tr, Xe_va, expr_pipe, n_comp = _fit_expression_block(df_tr, df_va, expr_cols)

    blocks_tr = [arr for arr in (Xc_tr, Xe_tr) if arr.shape[1] > 0]
    blocks_va = [arr for arr in (Xc_va, Xe_va) if arr.shape[1] > 0]
    if not blocks_tr:
        raise ValueError('No usable features remained after preprocessing.')

    X_tr = np.hstack(blocks_tr)
    X_va = np.hstack(blocks_va)
    if X_tr.shape[1] != X_va.shape[1]:
        raise ValueError('Feature dimension mismatch between train and validation transforms.')

    final_scaler = StandardScaler()
    X_tr = final_scaler.fit_transform(X_tr)
    X_va = final_scaler.transform(X_va)

    if not np.isfinite(X_tr).all() or not np.isfinite(X_va).all():
        raise ValueError('Non-finite values detected after encoding/scaling.')

    train_means = np.nanmean(X_tr, axis=0)
    train_stds = np.nanstd(X_tr, axis=0)
    checks: Dict[str, Any] = {
        'clinical_num_cols': num_cols,
        'clinical_cat_cols': cat_cols,
        'clinical_output_dim': int(Xc_tr.shape[1]),
        'expr_input_dim': int(len(expr_cols)),
        'expr_pca_components': int(n_comp),
        'expr_output_dim': int(Xe_tr.shape[1]),
        'final_dim': int(X_tr.shape[1]),
        'final_train_mean_abs_max': float(np.max(np.abs(train_means))),
        'final_train_std_min': float(np.min(train_stds)),
        'final_train_std_max': float(np.max(train_stds)),
        'final_constant_feature_count': int(np.sum(train_stds < 1e-12)),
    }

    return X_tr, X_va, clinical_pre, expr_pipe, final_scaler, checks


def transform_features(
    df: pd.DataFrame,
    clinical_cols: List[str],
    expr_cols: List[str],
    clinical_pre: Optional[ColumnTransformer],
    expr_pipe: Optional[Pipeline],
    final_scaler: StandardScaler,
) -> np.ndarray:
    blocks = []
    if clinical_cols:
        if clinical_pre is None:
            raise ValueError('Clinical preprocessor is missing but clinical features were requested.')
        blocks.append(np.asarray(clinical_pre.transform(df[clinical_cols].copy()), dtype=float))

    if expr_cols:
        if expr_pipe is None:
            raise ValueError('Expression preprocessor is missing but expression features were requested.')
        blocks.append(np.asarray(expr_pipe.transform(df[expr_cols].copy()), dtype=float))

    if not blocks:
        raise ValueError('No feature blocks available to transform.')

    X = np.hstack(blocks)
    X = final_scaler.transform(X)
    if not np.isfinite(X).all():
        raise ValueError('Non-finite values detected in transformed matrix.')
    return X


def run_cv_tuning(df_train: pd.DataFrame, clinical_cols: List[str], expr_cols: List[str]) -> pd.DataFrame:
    y = df_train['OS_EVENT'].to_numpy(dtype=int)
    class_counts = np.bincount(y, minlength=2)
    non_zero_counts = class_counts[class_counts > 0]
    if len(non_zero_counts) < 2:
        raise ValueError('Cross-validation requires both classes in training data.')

    n_splits = min(CV_FOLDS, int(np.min(non_zero_counts)))
    if n_splits < 2:
        raise ValueError(
            'Not enough minority-class samples for stratified cross-validation. '
            f'Minority count={int(np.min(non_zero_counts))}, required>=2.'
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for alpha in ALPHA_GRID:
        for l1_ratio in L1_GRID:
            fold_scores: List[float] = []
            failures: List[str] = []

            for fold_id, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(df_train)), y), start=1):
                tr = df_train.iloc[tr_idx]
                va = df_train.iloc[va_idx]

                try:
                    X_tr, X_va, _, _, _, _ = fit_transform_features(tr, va, clinical_cols, expr_cols)
                    t_tr = tr['OS_MONTHS'].to_numpy(dtype=float)
                    e_tr = tr['OS_EVENT'].to_numpy(dtype=int)
                    t_va = va['OS_MONTHS'].to_numpy(dtype=float)
                    e_va = va['OS_EVENT'].to_numpy(dtype=int)

                    model = CoxElasticNet(alpha=alpha, l1_ratio=l1_ratio, maxiter=MAXITER)
                    model.fit(X_tr, t_tr, e_tr)
                    if not bool(getattr(model, 'success_', True)):
                        failures.append(f'fold_{fold_id}:optimizer_not_converged')
                        continue

                    risk_va = model.predict_risk(X_va)
                    cidx = concordance_index_censored(t_va, e_va, risk_va)
                    if not np.isfinite(cidx):
                        failures.append(f'fold_{fold_id}:non_finite_c_index')
                        continue
                    fold_scores.append(float(cidx))
                except Exception as exc:
                    failures.append(f'fold_{fold_id}:{type(exc).__name__}')

            rows.append({
                'alpha': float(alpha),
                'l1_ratio': float(l1_ratio),
                'n_valid_folds': int(len(fold_scores)),
                'n_failed_folds': int(n_splits - len(fold_scores)),
                'mean_c_index': float(np.mean(fold_scores)) if fold_scores else np.nan,
                'std_c_index': float(np.std(fold_scores)) if fold_scores else np.nan,
                'fold_scores': ', '.join(f'{score:.4f}' for score in fold_scores),
                'failure_notes': '; '.join(failures),
            })

    cv_df = pd.DataFrame(rows)
    return cv_df.sort_values(
        ['n_valid_folds', 'mean_c_index', 'std_c_index'],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def select_best_hyperparameters(cv_df: pd.DataFrame) -> pd.Series:
    viable = cv_df[np.isfinite(cv_df['mean_c_index']) & (cv_df['n_valid_folds'] >= 2)].copy()
    if viable.empty:
        raise ValueError('No viable hyperparameter combination produced valid cross-validation results.')

    viable = viable.sort_values(
        ['n_valid_folds', 'mean_c_index', 'std_c_index'],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return viable.iloc[0]


def horizon_labels(time_arr: np.ndarray, event_arr: np.ndarray, horizon: float) -> Tuple[np.ndarray, np.ndarray]:
    pos = (event_arr == 1) & (time_arr <= horizon)
    neg = time_arr > horizon
    keep = pos | neg
    y = np.where(pos[keep], 1, 0)
    return keep, y


def time_dependent_conformal(
    r_cal: np.ndarray,
    t_cal: np.ndarray,
    e_cal: np.ndarray,
    r_te: np.ndarray,
    t_te: np.ndarray,
    e_te: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, Dict[str, Any]]]:
    rows_metrics: List[Dict[str, Any]] = []
    rows_pred: List[Dict[str, Any]] = []
    rows_cls: List[Dict[str, Any]] = []
    calibrators: Dict[int, Dict[str, Any]] = {}

    for h in HORIZONS_MONTHS:
        keep_cal, y_cal = horizon_labels(t_cal, e_cal, h)
        keep_te, y_te = horizon_labels(t_te, e_te, h)

        if len(y_cal) == 0 or len(y_te) == 0:
            rows_metrics.append({
                'horizon_months': h,
                'n_cal_known': int(len(y_cal)),
                'n_test_known': int(len(y_te)),
                'note': 'no_known_labels_after_censoring_filter',
            })
            continue

        if len(np.unique(y_cal)) < 2:
            rows_metrics.append({
                'horizon_months': h,
                'n_cal_known': int(len(y_cal)),
                'n_test_known': int(len(y_te)),
                'note': 'insufficient_calibration_class_variation',
            })
            continue

        if len(np.unique(y_te)) < 2:
            rows_metrics.append({
                'horizon_months': h,
                'n_cal_known': int(len(y_cal)),
                'n_test_known': int(len(y_te)),
                'note': 'insufficient_test_class_variation',
            })
            continue

        try:
            iso_h = IsotonicRegression(out_of_bounds='clip')
            p_cal = iso_h.fit_transform(r_cal[keep_cal], y_cal)
            p_te = iso_h.predict(r_te[keep_te])
        except Exception as exc:
            rows_metrics.append({
                'horizon_months': h,
                'n_cal_known': int(len(y_cal)),
                'n_test_known': int(len(y_te)),
                'note': f'isotonic_fit_failed:{type(exc).__name__}',
            })
            continue

        if not np.isfinite(p_cal).all() or not np.isfinite(p_te).all():
            rows_metrics.append({
                'horizon_months': h,
                'n_cal_known': int(len(y_cal)),
                'n_test_known': int(len(y_te)),
                'note': 'non_finite_probabilities_from_isotonic',
            })
            continue

        scores = np.abs(y_cal - p_cal)
        qhat_h = quantile_conformal(scores, CONFORMAL_ALPHA)
        if not np.isfinite(qhat_h):
            rows_metrics.append({
                'horizon_months': h,
                'n_cal_known': int(len(y_cal)),
                'n_test_known': int(len(y_te)),
                'note': 'non_finite_conformal_quantile',
            })
            continue

        p_lo = np.clip(p_te - qhat_h, 0.0, 1.0)
        p_hi = np.clip(p_te + qhat_h, 0.0, 1.0)
        coverage = float(np.mean((y_te >= p_lo) & (y_te <= p_hi)))

        rows_metrics.append({
            'horizon_months': h,
            'n_cal_known': int(len(y_cal)),
            'n_test_known': int(len(y_te)),
            'qhat': float(qhat_h),
            'coverage_known_test': coverage,
            'mean_interval_width': float(np.mean(p_hi - p_lo)),
            'note': 'ok',
        })

        # Threshold selection is based on calibration labels at each horizon.
        thr_grid = np.linspace(0, 1, 201)
        best_thr = 0.5
        best_j = -1.0
        for thr in thr_grid:
            yhat_c = (p_cal >= thr).astype(int)
            tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_cal, yhat_c, labels=[0, 1]).ravel()
            tpr_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
            tnr_c = tn_c / (tn_c + fp_c) if (tn_c + fp_c) > 0 else 0.0
            j_stat = tpr_c + tnr_c - 1.0
            if j_stat > best_j:
                best_j = j_stat
                best_thr = float(thr)

        yhat_t = (p_te >= best_thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, yhat_t, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        try:
            roc_auc = float(roc_auc_score(y_te, p_te))
        except ValueError:
            roc_auc = np.nan

        try:
            pr_auc = float(average_precision_score(y_te, p_te))
        except ValueError:
            pr_auc = np.nan

        rows_cls.append({
            'horizon_months': h,
            'n_test_known': int(len(y_te)),
            'event_rate_test_known': float(np.mean(y_te)),
            'threshold_from_calibration': best_thr,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'brier': float(brier_score_loss(y_te, p_te)),
            'accuracy': float(accuracy_score(y_te, yhat_t)),
            'precision': float(precision_score(y_te, yhat_t, zero_division=0)),
            'recall_sensitivity': float(recall_score(y_te, yhat_t, zero_division=0)),
            'specificity': float(specificity),
            'f1': float(f1_score(y_te, yhat_t, zero_division=0)),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
        })

        calibrators[int(h)] = {
            'isotonic': iso_h,
            'qhat': float(qhat_h),
            'threshold': float(best_thr),
        }

        idxs = np.where(keep_te)[0]
        for i_local, i_global in enumerate(idxs):
            rows_pred.append({
                'horizon_months': h,
                'test_row_index_within_split': int(i_global),
                'risk_score': float(r_te[i_global]),
                'y_true_by_horizon': int(y_te[i_local]),
                'p_event_hat': float(p_te[i_local]),
                'p_event_lo': float(p_lo[i_local]),
                'p_event_hi': float(p_hi[i_local]),
            })

    return pd.DataFrame(rows_metrics), pd.DataFrame(rows_pred), pd.DataFrame(rows_cls), calibrators


def month_interval_conformal(
    r_cal: np.ndarray,
    t_cal: np.ndarray,
    e_cal: np.ndarray,
    r_te: np.ndarray,
    t_te: np.ndarray,
    e_te: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float, Optional[IsotonicRegression], Dict[str, Any]]:
    """Conformal interval on months based on calibration events only."""
    lo = np.full(len(r_te), np.nan, dtype=float)
    hi = np.full(len(r_te), np.nan, dtype=float)

    cal_event_mask = e_cal == 1
    test_event_mask = e_te == 1
    n_cal_events = int(cal_event_mask.sum())
    n_test_events = int(test_event_mask.sum())

    details: Dict[str, Any] = {
        'n_calibration_events': n_cal_events,
        'n_test_events': n_test_events,
        'status': 'ok',
    }

    if n_cal_events < 2:
        details['status'] = 'skipped_insufficient_calibration_events'
        return lo, hi, np.nan, np.nan, None, details

    iso = IsotonicRegression(out_of_bounds='clip', increasing=False)
    x_cal = r_cal[cal_event_mask]
    y_cal = np.log1p(t_cal[cal_event_mask])

    try:
        iso.fit(x_cal, y_cal)
    except Exception as exc:
        details['status'] = f'skipped_isotonic_fit_failed:{type(exc).__name__}'
        return lo, hi, np.nan, np.nan, None, details

    yhat_cal = iso.predict(x_cal)
    resid_cal = np.abs(y_cal - yhat_cal)
    qhat = quantile_conformal(resid_cal, CONFORMAL_ALPHA)
    if not np.isfinite(qhat):
        details['status'] = 'skipped_non_finite_conformal_quantile'
        return lo, hi, np.nan, np.nan, None, details

    yhat_te = iso.predict(r_te)
    lo = np.clip(np.expm1(yhat_te - qhat), 0.0, None)
    hi = np.clip(np.expm1(yhat_te + qhat), 0.0, None)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)

    if n_test_events > 0:
        y_te_event = np.log1p(t_te[test_event_mask])
        lo_e = np.log1p(lo[test_event_mask])
        hi_e = np.log1p(hi[test_event_mask])
        coverage_event = float(np.mean((y_te_event >= lo_e) & (y_te_event <= hi_e)))
    else:
        coverage_event = np.nan

    return lo, hi, float(qhat), coverage_event, iso, details


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_dataframe().reset_index(drop=True)
    df_no, outlier_report = remove_outliers_iqr(df)

    idx_train, idx_cal, idx_test, split_report = split_three_way(df_no)
    clinical_cols, expr_cols = get_feature_sets(df_no)

    # Explicit leakage guard before any fitting.
    selected_features = set(clinical_cols) | set(expr_cols)
    forbidden_exact = {'OS_MONTHS', 'OS_EVENT', 'OS_STATUS', 'DFS_STATUS', 'DFS_EVENT'}
    forbidden_prefixes = ('OS_', 'DFS_')
    bad_features = sorted(
        c for c in selected_features
        if (c in forbidden_exact) or any(c.startswith(prefix) for prefix in forbidden_prefixes)
    )
    if bad_features:
        raise ValueError(f'Target-leakage features detected in model inputs: {bad_features}')

    df_train = df_no.iloc[idx_train].copy().reset_index(drop=True)
    cv_df = run_cv_tuning(df_train, clinical_cols, expr_cols)
    cv_df.to_csv(OUT_DIR / 'hyperparameter_cv_results.csv', index=False)

    best = select_best_hyperparameters(cv_df)
    best_alpha = float(best['alpha'])
    best_l1 = float(best['l1_ratio'])

    tr = df_no.iloc[idx_train].copy()
    cal = df_no.iloc[idx_cal].copy()
    te = df_no.iloc[idx_test].copy()

    X_tr, X_cal, clinical_pre, expr_pipe, final_scaler, feat_checks = fit_transform_features(
        tr,
        cal,
        clinical_cols,
        expr_cols,
    )
    X_te = transform_features(te, clinical_cols, expr_cols, clinical_pre, expr_pipe, final_scaler)

    t_tr = tr['OS_MONTHS'].to_numpy(dtype=float)
    e_tr = tr['OS_EVENT'].to_numpy(dtype=int)
    t_cal = cal['OS_MONTHS'].to_numpy(dtype=float)
    e_cal = cal['OS_EVENT'].to_numpy(dtype=int)
    t_te = te['OS_MONTHS'].to_numpy(dtype=float)
    e_te = te['OS_EVENT'].to_numpy(dtype=int)

    model = CoxElasticNet(alpha=best_alpha, l1_ratio=best_l1, maxiter=MAXITER)
    model.fit(X_tr, t_tr, e_tr)
    if not bool(getattr(model, 'success_', True)):
        raise RuntimeError(f'Cox optimization failed to converge: {getattr(model, "message_", "unknown")}')

    r_tr = model.predict_risk(X_tr)
    r_cal = model.predict_risk(X_cal)
    r_te = model.predict_risk(X_te)

    cidx_train = float(concordance_index_censored(t_tr, e_tr, r_tr))
    cidx_cal = float(concordance_index_censored(t_cal, e_cal, r_cal))
    cidx_test = float(concordance_index_censored(t_te, e_te, r_te))

    lo, hi, qhat, coverage_event, iso_months, month_conformal_info = month_interval_conformal(
        r_cal,
        t_cal,
        e_cal,
        r_te,
        t_te,
        e_te,
    )

    td_metrics, td_preds, td_cls, td_calibrators = time_dependent_conformal(
        r_cal,
        t_cal,
        e_cal,
        r_te,
        t_te,
        e_te,
    )
    td_metrics.to_csv(OUT_DIR / 'time_dependent_conformal_metrics.csv', index=False)
    td_preds.to_csv(OUT_DIR / 'time_dependent_conformal_test_predictions.csv', index=False)
    td_cls.to_csv(OUT_DIR / 'time_dependent_horizon_classification_metrics.csv', index=False)

    patient_ids = pd.concat([tr['PATIENT_ID'], cal['PATIENT_ID'], te['PATIENT_ID']], axis=0).astype(str).values
    pred_df = pd.DataFrame({
        'split': ['train'] * len(tr) + ['calibration'] * len(cal) + ['test'] * len(te),
        'PATIENT_ID': patient_ids,
        'OS_MONTHS': np.concatenate([t_tr, t_cal, t_te]),
        'OS_EVENT': np.concatenate([e_tr, e_cal, e_te]),
        'risk_score': np.concatenate([r_tr, r_cal, r_te]),
        'pred_months_lo_90': np.concatenate([np.full(len(tr), np.nan), np.full(len(cal), np.nan), lo]),
        'pred_months_hi_90': np.concatenate([np.full(len(tr), np.nan), np.full(len(cal), np.nan), hi]),
    })
    pred_df.to_csv(OUT_DIR / 'tuned_model_predictions.csv', index=False)

    coef_df = pd.DataFrame({'coef_index': np.arange(len(model.coef_)), 'coef_value': model.coef_})
    coef_df = coef_df.sort_values('coef_value', key=np.abs, ascending=False)
    coef_df.to_csv(OUT_DIR / 'tuned_model_coefficients.csv', index=False)

    interval_width = hi - lo
    finite_width = interval_width[np.isfinite(interval_width)]
    mean_width_months = float(np.mean(finite_width)) if len(finite_width) > 0 else np.nan

    consistency = {
        'input_file': str(INPUT_PATH),
        'rows_before_outlier_filter': int(len(df)),
        'rows_after_outlier_filter': int(len(df_no)),
        'outlier_report': outlier_report,
        'split_report': split_report,
        'feature_checks': feat_checks,
        'feature_counts': {
            'clinical_cols': int(len(clinical_cols)),
            'expr_cols': int(len(expr_cols)),
        },
        'leakage_check_passed': True,
    }
    (OUT_DIR / 'consistency_checks.json').write_text(json.dumps(consistency, indent=2), encoding='utf-8')

    metrics = {
        'input_file': str(INPUT_PATH),
        'best_params': {'alpha': best_alpha, 'l1_ratio': best_l1},
        'cv_best_mean_c_index': float(best['mean_c_index']),
        'cv_best_std_c_index': float(best['std_c_index']),
        'cv_best_valid_folds': int(best['n_valid_folds']),
        'n_total_after_outlier_filter': int(len(df_no)),
        'n_train': int(len(tr)),
        'n_calibration': int(len(cal)),
        'n_test': int(len(te)),
        'events_train': int(e_tr.sum()),
        'events_calibration': int(e_cal.sum()),
        'events_test': int(e_te.sum()),
        'cox_optimizer_success': bool(getattr(model, 'success_', True)),
        'cox_optimizer_message': str(getattr(model, 'message_', 'unknown')),
        'cox_optimizer_iterations': int(getattr(model, 'n_iter_', -1)),
        'c_index_train': cidx_train,
        'c_index_calibration': cidx_cal,
        'c_index_test': cidx_test,
        'conformal_month_interval': {
            'alpha': CONFORMAL_ALPHA,
            'qhat_log_months': qhat,
            'coverage_test_events_only': coverage_event,
            'mean_interval_width_months_test': mean_width_months,
            'status': month_conformal_info['status'],
            'n_calibration_events': month_conformal_info['n_calibration_events'],
            'n_test_events': month_conformal_info['n_test_events'],
        },
        'time_dependent_conformal': {
            'horizons_months': HORIZONS_MONTHS,
            'n_horizons_reported': int(td_metrics.shape[0]),
            'n_horizons_with_models': int(len(td_calibrators)),
        },
    }
    (OUT_DIR / 'tuned_model_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    artifact = {
        'clinical_cols': clinical_cols,
        'expr_cols': expr_cols,
        'clinical_pre': clinical_pre,
        'expr_pipe': expr_pipe,
        'final_scaler': final_scaler,
        'model': model,
        'month_interval_isotonic': iso_months,
        'month_interval_conformal_qhat': qhat,
        'month_interval_alpha': CONFORMAL_ALPHA,
        'time_dependent_calibrators': td_calibrators,
        'horizons_months': HORIZONS_MONTHS,
        'best_alpha': best_alpha,
        'best_l1_ratio': best_l1,
    }
    with open(OUT_DIR / 'final_locked_model.pkl', 'wb') as f:
        pickle.dump(artifact, f)

    readme = [
        '# Tuned Cox ENet + Outlier Filter + Time-dependent Conformal',
        '',
        f'Input: {INPUT_PATH}',
        f'Best alpha: {best_alpha}',
        f'Best l1_ratio: {best_l1}',
        f'CV best mean c-index: {float(best["mean_c_index"]):.4f}',
        f'Test c-index: {cidx_test:.4f}',
        f'Month-interval conformal status: {month_conformal_info["status"]}',
        f'Month-interval conformal coverage (event-only): {coverage_event:.4f}',
        '',
        'Files:',
        '- hyperparameter_cv_results.csv',
        '- tuned_model_metrics.json',
        '- tuned_model_predictions.csv',
        '- tuned_model_coefficients.csv',
        '- time_dependent_conformal_metrics.csv',
        '- time_dependent_conformal_test_predictions.csv',
        '- time_dependent_horizon_classification_metrics.csv',
        '- consistency_checks.json',
        '- final_locked_model.pkl',
    ]
    (OUT_DIR / 'README.md').write_text('\n'.join(readme), encoding='utf-8')

    print('Done.')
    print(f'Results written to: {OUT_DIR}')


if __name__ == '__main__':
    main()
