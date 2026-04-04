import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA


RANDOM_STATE = 42
ALPHA = 0.8
ENET_L1_RATIO = 0.3
SMOOTH_L1_EPS = 1e-6
MAXITER = 400
CONFORMAL_ALPHA = 0.10  # 90% intervals
EXPR_PCA_COMPONENTS = 50

BASE = Path('/home/illionar/Projects/ml_research')
INPUT_PATH = BASE / 'data' / 'preprocessed_cleaned' / 'patient_multiomic_cleaned.parquet'
OUT_DIR = BASE / 'data' / 'model_outputs' / 'cox_enet_conformal'


@dataclass
class CoxElasticNet:
    alpha: float = 0.8
    l1_ratio: float = 0.3
    smooth_l1_eps: float = 1e-6
    maxiter: int = 400

    def _nll_grad(self, beta: np.ndarray, X: np.ndarray, time: np.ndarray, event: np.ndarray) -> Tuple[float, np.ndarray]:
        # Sort descending so risk set for i is [:i+1]
        order = np.argsort(-time)
        Xo = X[order]
        eo = event[order]

        eta = np.clip(Xo @ beta, -40, 40)
        exp_eta = np.exp(eta)

        s0 = np.cumsum(exp_eta)
        s1 = np.cumsum(exp_eta[:, None] * Xo, axis=0)

        idx = np.where(eo == 1)[0]
        if len(idx) == 0:
            l1 = np.sqrt(beta * beta + self.smooth_l1_eps).sum()
            l2 = 0.5 * np.dot(beta, beta)
            nll = self.alpha * (self.l1_ratio * l1 + (1 - self.l1_ratio) * l2)
            grad_l1 = beta / np.sqrt(beta * beta + self.smooth_l1_eps)
            grad_l2 = beta
            grad = self.alpha * (self.l1_ratio * grad_l1 + (1 - self.l1_ratio) * grad_l2)
            return float(nll), grad

        loglik = np.sum(eta[idx] - np.log(s0[idx]))
        grad_loglik = np.sum(Xo[idx] - (s1[idx] / s0[idx, None]), axis=0)

        l1 = np.sqrt(beta * beta + self.smooth_l1_eps).sum()
        l2 = 0.5 * np.dot(beta, beta)

        nll = -loglik + self.alpha * (self.l1_ratio * l1 + (1 - self.l1_ratio) * l2)
        grad_l1 = beta / np.sqrt(beta * beta + self.smooth_l1_eps)
        grad_l2 = beta
        grad = -grad_loglik + self.alpha * (self.l1_ratio * grad_l1 + (1 - self.l1_ratio) * grad_l2)
        return float(nll), grad

    def fit(self, X: np.ndarray, time: np.ndarray, event: np.ndarray):
        beta0 = np.zeros(X.shape[1], dtype=float)
        res = minimize(
            fun=lambda b: self._nll_grad(b, X, time, event)[0],
            x0=beta0,
            jac=lambda b: self._nll_grad(b, X, time, event)[1],
            method='L-BFGS-B',
            options={'maxiter': self.maxiter},
        )
        self.coef_ = res.x
        self.success_ = bool(res.success)
        self.n_iter_ = int(res.nit)
        self.message_ = str(res.message)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_


def concordance_index_censored(times: np.ndarray, events: np.ndarray, risk_scores: np.ndarray) -> float:
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    risk_scores = np.asarray(risk_scores, dtype=float)

    t_i = times[:, None]
    t_j = times[None, :]
    e_i = events[:, None]

    comparable = (t_i < t_j) & (e_i == 1)
    denom = comparable.sum()
    if denom == 0:
        return np.nan

    s_i = risk_scores[:, None]
    s_j = risk_scores[None, :]
    conc = ((s_i > s_j) & comparable).sum()
    tied = ((s_i == s_j) & comparable).sum()
    return float((conc + 0.5 * tied) / denom)


def prepare_dataframe() -> pd.DataFrame:
    df = pd.read_parquet(INPUT_PATH)

    df['OS_MONTHS'] = pd.to_numeric(df.get('OS_MONTHS'), errors='coerce')
    df['OS_EVENT'] = pd.to_numeric(df.get('OS_EVENT'), errors='coerce')
    df = df.dropna(subset=['OS_MONTHS', 'OS_EVENT']).copy()
    df = df[df['OS_MONTHS'] > 0].copy()
    df['OS_EVENT'] = df['OS_EVENT'].astype(int)

    return df


def build_design_matrices(df: pd.DataFrame):
    # Based on ablation winner: Clinical + Expression
    clinical_candidates = ['AGE', 'SEX', 'RACE', 'ETHNICITY', 'CANCER_TYPE', 'AGE_GROUP']
    clinical_cols = [c for c in clinical_candidates if c in df.columns]
    expr_cols = [c for c in df.columns if c.startswith('EXPR_')]

    clinical_df = df[clinical_cols].copy()
    expr_df = df[expr_cols].copy()

    num_cols = clinical_df.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    cat_cols = [c for c in clinical_df.columns if c not in num_cols]

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

    expr_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='constant', fill_value=0.0)),
        ('sc', StandardScaler()),
        ('pca', PCA(n_components=min(EXPR_PCA_COMPONENTS, max(2, len(df) - 1), max(2, len(expr_cols))), random_state=RANDOM_STATE)),
    ])

    return clinical_df, expr_df, clinical_pre, expr_pipe


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


def quantile_conformal(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return np.nan
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(1.0, q_level)
    return float(np.quantile(scores, q_level, method='higher'))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_dataframe().reset_index(drop=True)
    clinical_df, expr_df, clinical_pre, expr_pipe = build_design_matrices(df)

    idx_train, idx_cal, idx_test = split_three_way(df)

    c_tr = clinical_df.iloc[idx_train]
    c_cal = clinical_df.iloc[idx_cal]
    c_te = clinical_df.iloc[idx_test]

    e_tr = expr_df.iloc[idx_train]
    e_cal = expr_df.iloc[idx_cal]
    e_te = expr_df.iloc[idx_test]

    Xc_tr = clinical_pre.fit_transform(c_tr)
    Xc_cal = clinical_pre.transform(c_cal)
    Xc_te = clinical_pre.transform(c_te)

    Xe_tr = expr_pipe.fit_transform(e_tr)
    Xe_cal = expr_pipe.transform(e_cal)
    Xe_te = expr_pipe.transform(e_te)

    X_tr = np.hstack([Xc_tr, Xe_tr])
    X_cal = np.hstack([Xc_cal, Xe_cal])
    X_te = np.hstack([Xc_te, Xe_te])

    final_scaler = StandardScaler()
    X_tr = final_scaler.fit_transform(X_tr)
    X_cal = final_scaler.transform(X_cal)
    X_te = final_scaler.transform(X_te)

    t_tr = df.loc[idx_train, 'OS_MONTHS'].values.astype(float)
    e_trgt = df.loc[idx_train, 'OS_EVENT'].values.astype(int)

    t_cal = df.loc[idx_cal, 'OS_MONTHS'].values.astype(float)
    e_calgt = df.loc[idx_cal, 'OS_EVENT'].values.astype(int)

    t_te = df.loc[idx_test, 'OS_MONTHS'].values.astype(float)
    e_tegt = df.loc[idx_test, 'OS_EVENT'].values.astype(int)

    model = CoxElasticNet(alpha=ALPHA, l1_ratio=ENET_L1_RATIO, smooth_l1_eps=SMOOTH_L1_EPS, maxiter=MAXITER)
    model.fit(X_tr, t_tr, e_trgt)

    r_tr = model.predict_risk(X_tr)
    r_cal = model.predict_risk(X_cal)
    r_te = model.predict_risk(X_te)

    cidx_train = concordance_index_censored(t_tr, e_trgt, r_tr)
    cidx_cal = concordance_index_censored(t_cal, e_calgt, r_cal)
    cidx_test = concordance_index_censored(t_te, e_tegt, r_te)

    # Conformal uncertainty on survival months via calibration events only.
    # Step 1: monotonic calibration model from risk -> log survival time.
    cal_event_mask = e_calgt == 1
    test_event_mask = e_tegt == 1

    iso = IsotonicRegression(out_of_bounds='clip', increasing=False)
    y_cal = np.log1p(t_cal[cal_event_mask])
    x_cal = r_cal[cal_event_mask]
    iso.fit(x_cal, y_cal)

    yhat_cal = iso.predict(x_cal)
    resid_cal = np.abs(y_cal - yhat_cal)
    qhat = quantile_conformal(resid_cal, CONFORMAL_ALPHA)

    yhat_te = iso.predict(r_te)
    lo = np.expm1(yhat_te - qhat)
    hi = np.expm1(yhat_te + qhat)
    lo = np.clip(lo, 0, None)

    # Empirical coverage on observed-event test points only
    y_te_event = np.log1p(t_te[test_event_mask])
    lo_e = np.log1p(lo[test_event_mask])
    hi_e = np.log1p(hi[test_event_mask])
    coverage_event = float(((y_te_event >= lo_e) & (y_te_event <= hi_e)).mean()) if len(y_te_event) else np.nan
    mean_interval_width_months = float(np.mean(hi - lo))

    # Save predictions
    pred = pd.DataFrame({
        'split': ['train'] * len(idx_train) + ['calibration'] * len(idx_cal) + ['test'] * len(idx_test),
        'row_index': np.concatenate([idx_train, idx_cal, idx_test]),
        'PATIENT_ID': pd.concat([df.loc[idx_train, 'PATIENT_ID'], df.loc[idx_cal, 'PATIENT_ID'], df.loc[idx_test, 'PATIENT_ID']]).values,
        'OS_MONTHS': np.concatenate([t_tr, t_cal, t_te]),
        'OS_EVENT': np.concatenate([e_trgt, e_calgt, e_tegt]),
        'risk_score': np.concatenate([r_tr, r_cal, r_te]),
        'pred_months_lo_90': np.concatenate([np.full(len(idx_train), np.nan), np.full(len(idx_cal), np.nan), lo]),
        'pred_months_hi_90': np.concatenate([np.full(len(idx_train), np.nan), np.full(len(idx_cal), np.nan), hi]),
    })
    pred.to_csv(OUT_DIR / 'cox_enet_conformal_predictions.csv', index=False)

    # Save coefficients
    coef_df = pd.DataFrame({
        'coef_index': np.arange(len(model.coef_)),
        'coef_value': model.coef_,
    }).sort_values('coef_value', key=np.abs, ascending=False)
    coef_df.to_csv(OUT_DIR / 'cox_enet_coefficients.csv', index=False)

    metrics = {
        'input_file': str(INPUT_PATH),
        'n_total': int(len(df)),
        'n_train': int(len(idx_train)),
        'n_calibration': int(len(idx_cal)),
        'n_test': int(len(idx_test)),
        'events_train': int(e_trgt.sum()),
        'events_calibration': int(e_calgt.sum()),
        'events_test': int(e_tegt.sum()),
        'cox_enet': {
            'alpha': ALPHA,
            'l1_ratio': ENET_L1_RATIO,
            'optimizer_success': model.success_,
            'optimizer_message': model.message_,
            'n_iter': model.n_iter_,
            'c_index_train': cidx_train,
            'c_index_calibration': cidx_cal,
            'c_index_test': cidx_test,
        },
        'conformal': {
            'alpha': CONFORMAL_ALPHA,
            'target_interval': f"{int((1-CONFORMAL_ALPHA)*100)}%",
            'qhat_abs_residual_log_months': qhat,
            'empirical_coverage_test_events_only': coverage_event,
            'mean_interval_width_months_test': mean_interval_width_months,
            'notes': 'Coverage is computed on uncensored test events only due to right-censoring constraints in split-conformal regression.',
        },
    }

    (OUT_DIR / 'cox_enet_conformal_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    summary_md = [
        '# Cox PH (Elastic Net) + Conformal Uncertainty',
        '',
        f"Input: {INPUT_PATH}",
        f"N={len(df)} (train={len(idx_train)}, calibration={len(idx_cal)}, test={len(idx_test)})",
        '',
        '## Cox Elastic Net',
        f"- alpha={ALPHA}",
        f"- l1_ratio={ENET_L1_RATIO}",
        f"- c-index train={cidx_train:.4f}",
        f"- c-index calibration={cidx_cal:.4f}",
        f"- c-index test={cidx_test:.4f}",
        '',
        '## Conformal Uncertainty (90% interval)',
        f"- qhat (abs residual on log-months)={qhat:.4f}",
        f"- empirical coverage on test events={coverage_event:.4f}",
        f"- mean interval width on test={mean_interval_width_months:.2f} months",
        '',
        '## Files',
        '- cox_enet_conformal_metrics.json',
        '- cox_enet_conformal_predictions.csv',
        '- cox_enet_coefficients.csv',
    ]

    (OUT_DIR / 'README.md').write_text('\n'.join(summary_md), encoding='utf-8')

    print('Done.')
    print(f'Results written to: {OUT_DIR}')


if __name__ == '__main__':
    main()
