import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.isotonic import IsotonicRegression

from train_cox_enet_conformal import (
    CoxElasticNet,
    concordance_index_censored,
    quantile_conformal,
)


RANDOM_STATE = 42
CONFORMAL_ALPHA = 0.10
EXPR_PCA_COMPONENTS = 50
MAXITER = 400

ALPHA_GRID = [0.1, 0.3, 0.8, 1.5, 3.0]
L1_GRID = [0.0, 0.1, 0.3, 0.5, 0.7]
CV_FOLDS = 5

BASE = Path('/home/illionar/Projects/ml_research')
INPUT_PATH = BASE / 'data' / 'preprocessed_cleaned' / 'patient_multiomic_cleaned.parquet'
OUT_DIR = BASE / 'data' / 'model_outputs' / 'cox_enet_conformal_tuned'


def prepare_dataframe() -> pd.DataFrame:
    df = pd.read_parquet(INPUT_PATH)
    df['OS_MONTHS'] = pd.to_numeric(df.get('OS_MONTHS'), errors='coerce')
    df['OS_EVENT'] = pd.to_numeric(df.get('OS_EVENT'), errors='coerce')
    df = df.dropna(subset=['OS_MONTHS', 'OS_EVENT']).copy()
    df = df[df['OS_MONTHS'] > 0].copy()
    df['OS_EVENT'] = df['OS_EVENT'].astype(int)
    return df


def split_three_way(df: pd.DataFrame):
    idx = np.arange(len(df))
    y = df['OS_EVENT'].values

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
    return idx_train, idx_cal, idx_test


def get_feature_sets(df: pd.DataFrame):
    clinical_candidates = ['AGE', 'SEX', 'RACE', 'ETHNICITY', 'CANCER_TYPE', 'AGE_GROUP']
    clinical_cols = [c for c in clinical_candidates if c in df.columns]
    expr_cols = [c for c in df.columns if c.startswith('EXPR_')]
    return clinical_cols, expr_cols


def fit_transform_features(df_tr: pd.DataFrame, df_va: pd.DataFrame, clinical_cols, expr_cols):
    c_tr = df_tr[clinical_cols].copy()
    c_va = df_va[clinical_cols].copy()
    e_tr = df_tr[expr_cols].copy()
    e_va = df_va[expr_cols].copy()

    num_cols = c_tr.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    cat_cols = [c for c in c_tr.columns if c not in num_cols]

    clinical_pre = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imp', SimpleImputer(strategy='median')),
                ('sc', StandardScaler()),
            ]), num_cols),
            ('cat', Pipeline([
                ('imp', SimpleImputer(strategy='most_frequent')),
                ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]), cat_cols),
        ],
        remainder='drop',
    )

    n_comp = min(EXPR_PCA_COMPONENTS, max(2, len(df_tr) - 1), max(2, len(expr_cols)))
    expr_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='constant', fill_value=0.0)),
        ('sc', StandardScaler()),
        ('pca', PCA(n_components=n_comp, random_state=RANDOM_STATE)),
    ])

    Xc_tr = clinical_pre.fit_transform(c_tr)
    Xc_va = clinical_pre.transform(c_va)

    Xe_tr = expr_pipe.fit_transform(e_tr)
    Xe_va = expr_pipe.transform(e_va)

    X_tr = np.hstack([Xc_tr, Xe_tr])
    X_va = np.hstack([Xc_va, Xe_va])

    final_scaler = StandardScaler()
    X_tr = final_scaler.fit_transform(X_tr)
    X_va = final_scaler.transform(X_va)

    return X_tr, X_va, clinical_pre, expr_pipe, final_scaler


def run_cv_tuning(df_train: pd.DataFrame, clinical_cols, expr_cols):
    y = df_train['OS_EVENT'].values
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for alpha in ALPHA_GRID:
        for l1_ratio in L1_GRID:
            fold_scores = []
            for tr_idx, va_idx in skf.split(np.zeros(len(df_train)), y):
                tr = df_train.iloc[tr_idx]
                va = df_train.iloc[va_idx]

                X_tr, X_va, _, _, _ = fit_transform_features(tr, va, clinical_cols, expr_cols)
                t_tr = tr['OS_MONTHS'].values.astype(float)
                e_tr = tr['OS_EVENT'].values.astype(int)
                t_va = va['OS_MONTHS'].values.astype(float)
                e_va = va['OS_EVENT'].values.astype(int)

                model = CoxElasticNet(alpha=alpha, l1_ratio=l1_ratio, maxiter=MAXITER)
                model.fit(X_tr, t_tr, e_tr)
                risk_va = model.predict_risk(X_va)
                cidx = concordance_index_censored(t_va, e_va, risk_va)
                fold_scores.append(cidx)

            rows.append({
                'alpha': alpha,
                'l1_ratio': l1_ratio,
                'mean_c_index': float(np.mean(fold_scores)),
                'std_c_index': float(np.std(fold_scores)),
                'fold_scores': ', '.join(f'{s:.4f}' for s in fold_scores),
            })

    cv_df = pd.DataFrame(rows).sort_values(['mean_c_index', 'std_c_index'], ascending=[False, True]).reset_index(drop=True)
    return cv_df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_dataframe().reset_index(drop=True)
    idx_train, idx_cal, idx_test = split_three_way(df)

    clinical_cols, expr_cols = get_feature_sets(df)

    df_train = df.iloc[idx_train].copy().reset_index(drop=True)
    cv_df = run_cv_tuning(df_train, clinical_cols, expr_cols)
    cv_df.to_csv(OUT_DIR / 'hyperparameter_cv_results.csv', index=False)

    best = cv_df.iloc[0]
    best_alpha = float(best['alpha'])
    best_l1 = float(best['l1_ratio'])

    # Fit final locked model on train split with best params
    tr = df.iloc[idx_train]
    cal = df.iloc[idx_cal]
    te = df.iloc[idx_test]

    X_tr, X_cal, clinical_pre, expr_pipe, final_scaler = fit_transform_features(tr, cal, clinical_cols, expr_cols)
    # transform test with same objects
    c_te = te[clinical_cols].copy()
    e_te = te[expr_cols].copy()
    Xc_te = clinical_pre.transform(c_te)
    Xe_te = expr_pipe.transform(e_te)
    X_te = np.hstack([Xc_te, Xe_te])
    X_te = final_scaler.transform(X_te)

    t_tr = tr['OS_MONTHS'].values.astype(float)
    e_tr = tr['OS_EVENT'].values.astype(int)
    t_cal = cal['OS_MONTHS'].values.astype(float)
    e_cal = cal['OS_EVENT'].values.astype(int)
    t_te = te['OS_MONTHS'].values.astype(float)
    e_te = te['OS_EVENT'].values.astype(int)

    model = CoxElasticNet(alpha=best_alpha, l1_ratio=best_l1, maxiter=MAXITER)
    model.fit(X_tr, t_tr, e_tr)

    r_tr = model.predict_risk(X_tr)
    r_cal = model.predict_risk(X_cal)
    r_te = model.predict_risk(X_te)

    cidx_train = concordance_index_censored(t_tr, e_tr, r_tr)
    cidx_cal = concordance_index_censored(t_cal, e_cal, r_cal)
    cidx_test = concordance_index_censored(t_te, e_te, r_te)

    # Conformal uncertainty on uncensored calibration events
    cal_event_mask = e_cal == 1
    test_event_mask = e_te == 1

    iso = IsotonicRegression(out_of_bounds='clip', increasing=False)
    x_cal = r_cal[cal_event_mask]
    y_cal = np.log1p(t_cal[cal_event_mask])
    iso.fit(x_cal, y_cal)

    yhat_cal = iso.predict(x_cal)
    resid_cal = np.abs(y_cal - yhat_cal)
    qhat = quantile_conformal(resid_cal, CONFORMAL_ALPHA)

    yhat_te = iso.predict(r_te)
    lo = np.expm1(yhat_te - qhat)
    hi = np.expm1(yhat_te + qhat)
    lo = np.clip(lo, 0, None)

    y_te_event = np.log1p(t_te[test_event_mask])
    lo_e = np.log1p(lo[test_event_mask])
    hi_e = np.log1p(hi[test_event_mask])
    coverage_event = float(((y_te_event >= lo_e) & (y_te_event <= hi_e)).mean()) if len(y_te_event) else np.nan

    pred_df = pd.DataFrame({
        'split': ['train'] * len(tr) + ['calibration'] * len(cal) + ['test'] * len(te),
        'PATIENT_ID': pd.concat([tr['PATIENT_ID'], cal['PATIENT_ID'], te['PATIENT_ID']]).values,
        'OS_MONTHS': np.concatenate([t_tr, t_cal, t_te]),
        'OS_EVENT': np.concatenate([e_tr, e_cal, e_te]),
        'risk_score': np.concatenate([r_tr, r_cal, r_te]),
        'pred_months_lo_90': np.concatenate([np.full(len(tr), np.nan), np.full(len(cal), np.nan), lo]),
        'pred_months_hi_90': np.concatenate([np.full(len(tr), np.nan), np.full(len(cal), np.nan), hi]),
    })
    pred_df.to_csv(OUT_DIR / 'tuned_model_predictions.csv', index=False)

    metrics = {
        'input_file': str(INPUT_PATH),
        'best_params': {'alpha': best_alpha, 'l1_ratio': best_l1},
        'cv_best_mean_c_index': float(best['mean_c_index']),
        'cv_best_std_c_index': float(best['std_c_index']),
        'n_total': int(len(df)),
        'n_train': int(len(tr)),
        'n_calibration': int(len(cal)),
        'n_test': int(len(te)),
        'events_train': int(e_tr.sum()),
        'events_calibration': int(e_cal.sum()),
        'events_test': int(e_te.sum()),
        'c_index_train': cidx_train,
        'c_index_calibration': cidx_cal,
        'c_index_test': cidx_test,
        'conformal_alpha': CONFORMAL_ALPHA,
        'conformal_qhat_log_months': qhat,
        'conformal_coverage_test_events_only': coverage_event,
        'conformal_mean_interval_width_months_test': float(np.mean(hi - lo)),
    }
    (OUT_DIR / 'tuned_model_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    coef_df = pd.DataFrame({'coef_index': np.arange(len(model.coef_)), 'coef_value': model.coef_})
    coef_df = coef_df.sort_values('coef_value', key=np.abs, ascending=False)
    coef_df.to_csv(OUT_DIR / 'tuned_model_coefficients.csv', index=False)

    artifact = {
        'clinical_cols': clinical_cols,
        'expr_cols': expr_cols,
        'clinical_pre': clinical_pre,
        'expr_pipe': expr_pipe,
        'final_scaler': final_scaler,
        'model': model,
        'isotonic': iso,
        'conformal_qhat': qhat,
        'conformal_alpha': CONFORMAL_ALPHA,
        'best_alpha': best_alpha,
        'best_l1_ratio': best_l1,
    }
    with open(OUT_DIR / 'final_locked_model.pkl', 'wb') as f:
        pickle.dump(artifact, f)

    readme = [
        '# Tuned Cox ENet + Conformal',
        '',
        f"Input: {INPUT_PATH}",
        f"Best alpha: {best_alpha}",
        f"Best l1_ratio: {best_l1}",
        f"CV best mean c-index: {float(best['mean_c_index']):.4f}",
        f"Test c-index: {cidx_test:.4f}",
        f"Conformal 90% coverage on uncensored test events: {coverage_event:.4f}",
        '',
        'Artifacts:',
        '- hyperparameter_cv_results.csv',
        '- tuned_model_metrics.json',
        '- tuned_model_predictions.csv',
        '- tuned_model_coefficients.csv',
        '- final_locked_model.pkl',
    ]
    (OUT_DIR / 'README.md').write_text('\n'.join(readme), encoding='utf-8')

    print('Done.')
    print(f'Results written to: {OUT_DIR}')


if __name__ == '__main__':
    main()
