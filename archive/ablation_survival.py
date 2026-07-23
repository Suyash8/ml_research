import os
import json
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, List, Tuple
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA


RANDOM_STATE = 42
N_SPLITS = 5
RIDGE_ALPHA = 1.0
EXPR_PCA_COMPONENTS = 30
CNA_PCA_COMPONENTS = 30

DATA_DIR = os.path.join("/home/illionar/Projects/ml_research", "data", "preprocessed")
PARQUET_PATH = os.path.join(DATA_DIR, "patient_multiomic.parquet")
CSV_FALLBACK_PATH = os.path.join(DATA_DIR, "patient_multiomic.csv")


def load_multiomic() -> pd.DataFrame:
    if os.path.exists(PARQUET_PATH):
        df = pd.read_parquet(PARQUET_PATH)
        source = PARQUET_PATH
    elif os.path.exists(CSV_FALLBACK_PATH):
        df = pd.read_csv(CSV_FALLBACK_PATH, low_memory=False)
        source = CSV_FALLBACK_PATH
    else:
        raise FileNotFoundError("No patient_multiomic.{parquet,csv} found in data/preprocessed")

    print(f"Loaded: {source}")
    print(f"Shape: {df.shape}")
    return df


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

    concordant = ((s_i > s_j) & comparable).sum()
    tied = ((s_i == s_j) & comparable).sum()
    return float((concordant + 0.5 * tied) / denom)


@dataclass
class CoxRidgeModel:
    alpha: float = 1.0
    maxiter: int = 300

    def fit(self, X: np.ndarray, times: np.ndarray, events: np.ndarray):
        X = np.asarray(X, dtype=float)
        times = np.asarray(times, dtype=float)
        events = np.asarray(events, dtype=int)

        order = np.argsort(-times)  # descending by time
        Xo = X[order]
        to = times[order]
        eo = events[order]

        n_features = Xo.shape[1]

        def nll_and_grad(beta: np.ndarray) -> Tuple[float, np.ndarray]:
            eta = Xo @ beta
            eta = np.clip(eta, -50, 50)
            exp_eta = np.exp(eta)

            # Risk-set cumulative sums in descending-time order
            s0 = np.cumsum(exp_eta)
            s1 = np.cumsum(exp_eta[:, None] * Xo, axis=0)

            event_idx = np.where(eo == 1)[0]
            if len(event_idx) == 0:
                nll = 0.5 * self.alpha * np.dot(beta, beta)
                grad = self.alpha * beta
                return nll, grad

            loglik = np.sum(eta[event_idx] - np.log(s0[event_idx]))
            grad_loglik = np.sum(Xo[event_idx] - (s1[event_idx] / s0[event_idx, None]), axis=0)

            nll = -loglik + 0.5 * self.alpha * np.dot(beta, beta)
            grad = -grad_loglik + self.alpha * beta
            return float(nll), grad

        beta0 = np.zeros(n_features, dtype=float)

        res = minimize(
            fun=lambda b: nll_and_grad(b)[0],
            x0=beta0,
            jac=lambda b: nll_and_grad(b)[1],
            method="L-BFGS-B",
            options={"maxiter": self.maxiter, "disp": False},
        )

        if not res.success:
            print(f"Warning: optimizer did not fully converge: {res.message}")

        self.coef_ = res.x
        self.n_iter_ = res.nit
        self.success_ = res.success
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_


def build_feature_blocks(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    target_cols = {"OS_MONTHS", "OS_EVENT"}

    expr_cols = [c for c in df.columns if c.startswith("EXPR_")]
    cna_cols = [c for c in df.columns if c.startswith("CNA_")]

    clinical_candidate_cols = [
        "AGE",
        "SEX",
        "RACE",
        "ETHNICITY",
        "CANCER_TYPE",
        "IS_FALLBACK",
        "AGE_GROUP",
    ]
    clinical_cols = [c for c in clinical_candidate_cols if c in df.columns]

    mut_cols = ["MUTATION_BURDEN"] if "MUTATION_BURDEN" in df.columns else []

    # Keep only rows with valid OS target
    work = df.copy()
    work["OS_MONTHS"] = pd.to_numeric(work["OS_MONTHS"], errors="coerce")
    work["OS_EVENT"] = pd.to_numeric(work["OS_EVENT"], errors="coerce")
    work = work.dropna(subset=["OS_MONTHS", "OS_EVENT"])
    work = work[work["OS_MONTHS"] > 0].copy()
    work["OS_EVENT"] = work["OS_EVENT"].astype(int)

    print(f"Rows kept for survival endpoint (OS): {len(work)}")
    print(f"Events: {work['OS_EVENT'].sum()} | Censored: {(1 - work['OS_EVENT']).sum()}")

    blocks = {
        "clinical": work[clinical_cols].copy(),
        "mutation": work[mut_cols].copy() if mut_cols else pd.DataFrame(index=work.index),
        "expression": work[expr_cols].copy(),
        "cna": work[cna_cols].copy(),
        "time": work["OS_MONTHS"].astype(float).values,
        "event": work["OS_EVENT"].astype(int).values,
        "strata": (
            work["CANCER_TYPE"].astype(str).fillna("UNK")
            + "_"
            + work["OS_EVENT"].astype(str)
        ).values,
    }

    print(f"Feature groups -> clinical: {blocks['clinical'].shape[1]}, mutation: {blocks['mutation'].shape[1]}, expression: {blocks['expression'].shape[1]}, cna: {blocks['cna'].shape[1]}")
    return blocks


def transform_fold(
    clinical_train: pd.DataFrame,
    clinical_test: pd.DataFrame,
    mutation_train: pd.DataFrame,
    mutation_test: pd.DataFrame,
    expr_train: pd.DataFrame,
    expr_test: pd.DataFrame,
    cna_train: pd.DataFrame,
    cna_test: pd.DataFrame,
):
    # Clinical preprocessing
    num_cols = clinical_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in clinical_train.columns if c not in num_cols]

    clinical_pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    Xc_tr = clinical_pre.fit_transform(clinical_train) if clinical_train.shape[1] else np.empty((len(clinical_train), 0))
    Xc_te = clinical_pre.transform(clinical_test) if clinical_test.shape[1] else np.empty((len(clinical_test), 0))

    # Mutation preprocessing
    if mutation_train.shape[1]:
        imp_mut = SimpleImputer(strategy="median")
        Xm_tr = imp_mut.fit_transform(mutation_train)
        Xm_te = imp_mut.transform(mutation_test)
        sc_mut = StandardScaler()
        Xm_tr = sc_mut.fit_transform(Xm_tr)
        Xm_te = sc_mut.transform(Xm_te)
    else:
        Xm_tr = np.empty((len(mutation_train), 0))
        Xm_te = np.empty((len(mutation_test), 0))

    # Expression -> PCA
    if expr_train.shape[1]:
        imp_expr = SimpleImputer(strategy="constant", fill_value=0.0)
        Xe_tr_raw = imp_expr.fit_transform(expr_train)
        Xe_te_raw = imp_expr.transform(expr_test)

        sc_expr = StandardScaler(with_mean=True, with_std=True)
        Xe_tr_std = sc_expr.fit_transform(Xe_tr_raw)
        Xe_te_std = sc_expr.transform(Xe_te_raw)

        n_expr = min(EXPR_PCA_COMPONENTS, Xe_tr_std.shape[0] - 1, Xe_tr_std.shape[1])
        if n_expr > 0:
            pca_expr = PCA(n_components=n_expr, random_state=RANDOM_STATE)
            Xe_tr = pca_expr.fit_transform(Xe_tr_std)
            Xe_te = pca_expr.transform(Xe_te_std)
        else:
            Xe_tr = np.empty((len(expr_train), 0))
            Xe_te = np.empty((len(expr_test), 0))
    else:
        Xe_tr = np.empty((len(expr_train), 0))
        Xe_te = np.empty((len(expr_test), 0))

    # CNA -> PCA
    if cna_train.shape[1]:
        imp_cna = SimpleImputer(strategy="constant", fill_value=0.0)
        Xn_tr_raw = imp_cna.fit_transform(cna_train)
        Xn_te_raw = imp_cna.transform(cna_test)

        sc_cna = StandardScaler(with_mean=True, with_std=True)
        Xn_tr_std = sc_cna.fit_transform(Xn_tr_raw)
        Xn_te_std = sc_cna.transform(Xn_te_raw)

        n_cna = min(CNA_PCA_COMPONENTS, Xn_tr_std.shape[0] - 1, Xn_tr_std.shape[1])
        if n_cna > 0:
            pca_cna = PCA(n_components=n_cna, random_state=RANDOM_STATE)
            Xn_tr = pca_cna.fit_transform(Xn_tr_std)
            Xn_te = pca_cna.transform(Xn_te_std)
        else:
            Xn_tr = np.empty((len(cna_train), 0))
            Xn_te = np.empty((len(cna_test), 0))
    else:
        Xn_tr = np.empty((len(cna_train), 0))
        Xn_te = np.empty((len(cna_test), 0))

    return Xc_tr, Xc_te, Xm_tr, Xm_te, Xe_tr, Xe_te, Xn_tr, Xn_te


def run_ablation(blocks: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    time = blocks["time"]
    event = blocks["event"]
    strata = blocks["strata"]

    clinical = blocks["clinical"]
    mutation = blocks["mutation"]
    expression = blocks["expression"]
    cna = blocks["cna"]

    experiments = {
        "Clinical only": ["clinical"],
        "Clinical + Mutation": ["clinical", "mutation"],
        "Clinical + Expression(PCA)": ["clinical", "expression"],
        "Clinical + CNA(PCA)": ["clinical", "cna"],
        "Clinical + Expr + CNA": ["clinical", "expression", "cna"],
        "All (Clinical + Mutation + Expr + CNA)": ["clinical", "mutation", "expression", "cna"],
        "Omics only (Mutation + Expr + CNA)": ["mutation", "expression", "cna"],
    }

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    rows = []

    for exp_name, includes in experiments.items():
        fold_scores = []

        for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(time)), strata), start=1):
            c_tr, c_te = clinical.iloc[tr_idx], clinical.iloc[te_idx]
            m_tr, m_te = mutation.iloc[tr_idx], mutation.iloc[te_idx]
            e_tr, e_te = expression.iloc[tr_idx], expression.iloc[te_idx]
            n_tr, n_te = cna.iloc[tr_idx], cna.iloc[te_idx]

            Xc_tr, Xc_te, Xm_tr, Xm_te, Xe_tr, Xe_te, Xn_tr, Xn_te = transform_fold(
                c_tr, c_te, m_tr, m_te, e_tr, e_te, n_tr, n_te
            )

            block_map_tr = {
                "clinical": Xc_tr,
                "mutation": Xm_tr,
                "expression": Xe_tr,
                "cna": Xn_tr,
            }
            block_map_te = {
                "clinical": Xc_te,
                "mutation": Xm_te,
                "expression": Xe_te,
                "cna": Xn_te,
            }

            Xtr = np.hstack([block_map_tr[k] for k in includes if block_map_tr[k].shape[1] > 0])
            Xte = np.hstack([block_map_te[k] for k in includes if block_map_te[k].shape[1] > 0])

            # Final scaling before Cox fit
            sc_final = StandardScaler(with_mean=True, with_std=True)
            Xtr = sc_final.fit_transform(Xtr)
            Xte = sc_final.transform(Xte)

            model = CoxRidgeModel(alpha=RIDGE_ALPHA, maxiter=300)
            model.fit(Xtr, time[tr_idx], event[tr_idx])
            risk = model.predict_risk(Xte)
            cidx = concordance_index_censored(time[te_idx], event[te_idx], risk)
            fold_scores.append(cidx)

            print(f"[{exp_name}] fold {fold}/{N_SPLITS} c-index = {cidx:.4f}  (features={Xtr.shape[1]})")

        rows.append(
            {
                "experiment": exp_name,
                "mean_c_index": float(np.nanmean(fold_scores)),
                "std_c_index": float(np.nanstd(fold_scores)),
                "fold_scores": ", ".join(f"{s:.4f}" for s in fold_scores),
            }
        )

    res = pd.DataFrame(rows).sort_values("mean_c_index", ascending=False).reset_index(drop=True)
    return res


def main():
    df = load_multiomic()
    blocks = build_feature_blocks(df)
    results = run_ablation(blocks)

    out_csv = os.path.join(DATA_DIR, "ablation_survival_results.csv")
    out_json = os.path.join(DATA_DIR, "ablation_survival_results.json")

    results.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(results.to_dict(orient="records"), f, indent=2)

    print("\n=== Ablation Summary (sorted) ===")
    print(results.to_string(index=False))
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
