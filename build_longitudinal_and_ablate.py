import os
import json
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, List, Tuple
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = "/home/illionar/Projects/ml_research"
DATA_DIR = os.path.join(BASE_DIR, "data")
PREP_DIR = os.path.join(DATA_DIR, "preprocessed")
CANCER_TYPES = ["gbm_tcga", "lihc_tcga", "paad_tcga", "skcm_tcga"]
CANCER_LABELS = {
    "gbm_tcga": "GBM",
    "lihc_tcga": "LIHC",
    "paad_tcga": "PAAD",
    "skcm_tcga": "SKCM",
}

TOP_N_GENES = 300
RANDOM_STATE = 42
N_SPLITS = 5
RIDGE_ALPHA = 1.0
EXPR_PCA_COMPONENTS = 30
CNA_PCA_COMPONENTS = 30


def load_clinical(filepath: str) -> pd.DataFrame:
    with open(filepath) as f:
        skip = 0
        for line in f:
            if line.startswith("#"):
                skip += 1
            else:
                break
    return pd.read_csv(
        filepath,
        sep="\t",
        skiprows=skip,
        na_values=["[Not Available]", "[Not Applicable]", "[Not Evaluated]", "[Pending]"],
        low_memory=False,
    )


def norm_sample_id(s: str) -> str:
    if pd.isna(s):
        return np.nan
    parts = str(s).upper().split("-")
    return "-".join(parts[:4]) if len(parts) >= 4 else str(s).upper()


def load_patient_table() -> pd.DataFrame:
    # Prefer cleaned table from notebook pipeline
    clinical_path = os.path.join(PREP_DIR, "clinical_all_clean.parquet")
    if os.path.exists(clinical_path):
        c = pd.read_parquet(clinical_path)
    else:
        # Fallback: rebuild quickly from raw patient files
        pieces = []
        for ct in CANCER_TYPES:
            fp = os.path.join(DATA_DIR, ct, "data_clinical_patient.txt")
            d = load_clinical(fp)
            d["CANCER_TYPE"] = CANCER_LABELS[ct]
            pieces.append(d)
        c = pd.concat(pieces, ignore_index=True)

    c = c.copy()
    c["OS_MONTHS"] = pd.to_numeric(c.get("OS_MONTHS"), errors="coerce")

    if "OS_EVENT" not in c.columns and "OS_STATUS" in c.columns:
        c["OS_EVENT"] = c["OS_STATUS"].astype(str).str.upper().str.contains("1:|DECEASED", regex=True).astype(int)

    c["OS_EVENT"] = pd.to_numeric(c.get("OS_EVENT"), errors="coerce")
    c = c.dropna(subset=["PATIENT_ID", "OS_MONTHS", "OS_EVENT"]).copy()
    c = c[c["OS_MONTHS"] > 0].copy()
    c["OS_EVENT"] = c["OS_EVENT"].astype(int)
    c["OS_DAYS"] = c["OS_MONTHS"] * 30.4375

    keep_cols = [
        col for col in ["PATIENT_ID", "CANCER_TYPE", "AGE", "SEX", "RACE", "ETHNICITY", "AGE_GROUP", "OS_MONTHS", "OS_DAYS", "OS_EVENT"]
        if col in c.columns
    ]

    c = c[keep_cols].drop_duplicates(subset=["PATIENT_ID", "CANCER_TYPE"], keep="first")
    return c


def load_sample_metadata() -> pd.DataFrame:
    pieces = []
    for ct in CANCER_TYPES:
        s = load_clinical(os.path.join(DATA_DIR, ct, "data_clinical_sample.txt"))
        s["CANCER_TYPE"] = CANCER_LABELS[ct]
        s = s[[c for c in s.columns if c in [
            "PATIENT_ID", "SAMPLE_ID", "DAYS_TO_COLLECTION", "DAYS_TO_SPECIMEN_COLLECTION", "SAMPLE_TYPE", "SAMPLE_TYPE_ID", "CANCER_TYPE"
        ]]].copy()
        pieces.append(s)

    samp = pd.concat(pieces, ignore_index=True)
    samp["SAMPLE_ID_NORM"] = samp["SAMPLE_ID"].apply(norm_sample_id)

    d1 = pd.to_numeric(samp.get("DAYS_TO_COLLECTION"), errors="coerce")
    d2 = pd.to_numeric(samp.get("DAYS_TO_SPECIMEN_COLLECTION"), errors="coerce")
    samp["SAMPLE_DAY"] = d1.fillna(d2)
    samp["SAMPLE_DAY"] = samp["SAMPLE_DAY"].fillna(0).clip(lower=0)

    samp["SAMPLE_TYPE_CODE"] = (
        samp["SAMPLE_ID"].astype(str).str.split("-").str[3].fillna("00")
    )
    samp["IS_PRIMARY"] = samp["SAMPLE_TYPE_CODE"].eq("01")

    return samp


def load_mutation_burden_per_sample() -> pd.DataFrame:
    out = []
    for ct in CANCER_TYPES:
        print(f"Loading mutations for {ct} ...")
        m = pd.read_csv(
            os.path.join(DATA_DIR, ct, "data_mutations.txt"),
            sep="\t",
            usecols=["Tumor_Sample_Barcode"],
            low_memory=False,
        )
        if "Tumor_Sample_Barcode" not in m.columns:
            continue
        b = m.groupby("Tumor_Sample_Barcode").size().rename("MUTATION_BURDEN").reset_index()
        b["SAMPLE_ID_NORM"] = b["Tumor_Sample_Barcode"].apply(norm_sample_id)
        b["CANCER_TYPE"] = CANCER_LABELS[ct]
        out.append(b[["SAMPLE_ID_NORM", "CANCER_TYPE", "MUTATION_BURDEN"]])

    if not out:
        return pd.DataFrame(columns=["SAMPLE_ID_NORM", "CANCER_TYPE", "MUTATION_BURDEN"])

    b = pd.concat(out, ignore_index=True)
    b = b.groupby(["SAMPLE_ID_NORM", "CANCER_TYPE"], as_index=False)["MUTATION_BURDEN"].sum()
    return b


def load_expression_and_cna_top_genes(sample_norm_index: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    expr_rows = []
    cna_rows = []

    # Build per-sample rows across cohorts
    for ct in CANCER_TYPES:
        label = CANCER_LABELS[ct]
        cohort_samples = set(sample_norm_index.loc[sample_norm_index["CANCER_TYPE"] == label, "SAMPLE_ID_NORM"].dropna().unique().tolist())

        expr = pd.read_csv(os.path.join(DATA_DIR, ct, "data_mrna_seq_v2_rsem.txt"), sep="\t", low_memory=False)
        expr = expr.drop(columns=["Entrez_Gene_Id"], errors="ignore").set_index("Hugo_Symbol")
        expr = expr[~expr.index.duplicated(keep="first")]
        expr.columns = [norm_sample_id(c) for c in expr.columns]
        keep_cols_expr = [c for c in expr.columns if c in cohort_samples]
        if keep_cols_expr:
            e = expr[keep_cols_expr].T
            e["SAMPLE_ID_NORM"] = e.index
            e["CANCER_TYPE"] = label
            expr_rows.append(e.reset_index(drop=True))

        cna = pd.read_csv(os.path.join(DATA_DIR, ct, "data_cna.txt"), sep="\t", low_memory=False)
        cna = cna.drop(columns=["Entrez_Gene_Id"], errors="ignore").set_index("Hugo_Symbol")
        cna = cna[~cna.index.duplicated(keep="first")]
        cna.columns = [norm_sample_id(c) for c in cna.columns]
        keep_cols_cna = [c for c in cna.columns if c in cohort_samples]
        if keep_cols_cna:
            n = cna[keep_cols_cna].T
            n["SAMPLE_ID_NORM"] = n.index
            n["CANCER_TYPE"] = label
            cna_rows.append(n.reset_index(drop=True))

    expr_all = pd.concat(expr_rows, ignore_index=True)
    cna_all = pd.concat(cna_rows, ignore_index=True)

    gene_cols_expr = [c for c in expr_all.columns if c not in ["SAMPLE_ID_NORM", "CANCER_TYPE"]]
    expr_num = expr_all[gene_cols_expr].apply(pd.to_numeric, errors="coerce")
    gene_var = expr_num.var(axis=0, skipna=True).sort_values(ascending=False)
    top_genes = gene_var.head(TOP_N_GENES).index.tolist()

    expr_top = expr_all[["SAMPLE_ID_NORM", "CANCER_TYPE"] + top_genes].copy()
    expr_top[top_genes] = expr_top[top_genes].apply(pd.to_numeric, errors="coerce")
    expr_top[top_genes] = np.log2(expr_top[top_genes].fillna(0) + 1)
    expr_top = expr_top.rename(columns={g: f"EXPR_{g}" for g in top_genes})

    cna_genes_present = [g for g in top_genes if g in cna_all.columns]
    cna_top = cna_all[["SAMPLE_ID_NORM", "CANCER_TYPE"] + cna_genes_present].copy()
    cna_top[cna_genes_present] = cna_top[cna_genes_present].apply(pd.to_numeric, errors="coerce")
    for g in top_genes:
        if g not in cna_top.columns:
            cna_top[g] = 0.0
    cna_top = cna_top[["SAMPLE_ID_NORM", "CANCER_TYPE"] + top_genes]
    cna_top = cna_top.rename(columns={g: f"CNA_{g}" for g in top_genes})

    return expr_top, cna_top, top_genes


def create_sample_omic_long() -> pd.DataFrame:
    patients = load_patient_table()
    sample_meta = load_sample_metadata()

    # Keep only patients with survival endpoints
    sm = sample_meta.merge(
        patients[["PATIENT_ID", "CANCER_TYPE", "OS_DAYS", "OS_EVENT"]],
        on=["PATIENT_ID", "CANCER_TYPE"],
        how="inner",
    )

    # Remove obviously post-outcome samples
    sm = sm[sm["SAMPLE_DAY"] <= sm["OS_DAYS"] + 1e-6].copy()

    mut = load_mutation_burden_per_sample()
    expr_top, cna_top, top_genes = load_expression_and_cna_top_genes(sm[["SAMPLE_ID_NORM", "CANCER_TYPE"]])

    out = sm.merge(mut, on=["SAMPLE_ID_NORM", "CANCER_TYPE"], how="left")
    out = out.merge(expr_top, on=["SAMPLE_ID_NORM", "CANCER_TYPE"], how="left")
    out = out.merge(cna_top, on=["SAMPLE_ID_NORM", "CANCER_TYPE"], how="left")

    out["MUTATION_BURDEN"] = pd.to_numeric(out["MUTATION_BURDEN"], errors="coerce").fillna(0)

    # De-duplicate same patient/sample entries
    out = out.sort_values(["PATIENT_ID", "SAMPLE_DAY", "SAMPLE_ID"]).drop_duplicates(
        subset=["PATIENT_ID", "SAMPLE_ID_NORM", "CANCER_TYPE"], keep="first"
    )

    out_path = os.path.join(PREP_DIR, "sample_omic_long.parquet")
    out.to_parquet(out_path, index=False)

    print(f"Saved sample-level longitudinal omics: {out_path}")
    print(f"Shape: {out.shape}")
    print(f"Top genes used: {len(top_genes)}")
    return out


def create_survival_intervals(sample_long: pd.DataFrame) -> pd.DataFrame:
    patients = load_patient_table()
    clinical_cols = [c for c in ["AGE", "SEX", "RACE", "ETHNICITY", "AGE_GROUP"] if c in patients.columns]

    omic_cols = [c for c in sample_long.columns if c.startswith("EXPR_") or c.startswith("CNA_")]

    records = []

    sample_long = sample_long.sort_values(["PATIENT_ID", "SAMPLE_DAY", "SAMPLE_ID"]).copy()

    for _, prow in patients.iterrows():
        pid = prow["PATIENT_ID"]
        ct = prow["CANCER_TYPE"]
        os_days = float(prow["OS_DAYS"])
        os_event = int(prow["OS_EVENT"])

        sub = sample_long[(sample_long["PATIENT_ID"] == pid) & (sample_long["CANCER_TYPE"] == ct)].copy()

        # If no sample row exists, create one synthetic baseline row
        if sub.empty:
            sub = pd.DataFrame({
                "PATIENT_ID": [pid],
                "CANCER_TYPE": [ct],
                "SAMPLE_ID": [np.nan],
                "SAMPLE_ID_NORM": [np.nan],
                "SAMPLE_DAY": [0.0],
                "SAMPLE_TYPE_CODE": ["00"],
                "IS_PRIMARY": [False],
                "MUTATION_BURDEN": [0.0],
            })
            for c in omic_cols:
                sub[c] = np.nan

        sub = sub.sort_values(["SAMPLE_DAY", "SAMPLE_ID_NORM"], na_position="last")
        # one row per unique day, keep first
        sub = sub.drop_duplicates(subset=["SAMPLE_DAY"], keep="first")

        # Ensure day 0 anchor exists (LOCF starts at baseline)
        if (sub["SAMPLE_DAY"] > 0).all():
            baseline = sub.iloc[[0]].copy()
            baseline["SAMPLE_DAY"] = 0.0
            baseline["SAMPLE_ID"] = np.nan
            baseline["SAMPLE_ID_NORM"] = np.nan
            baseline["SAMPLE_TYPE_CODE"] = "00"
            baseline["IS_PRIMARY"] = False
            sub = pd.concat([baseline, sub], ignore_index=True).sort_values("SAMPLE_DAY")

        times = sub["SAMPLE_DAY"].astype(float).values
        times = np.clip(times, 0, max(os_days - 1e-6, 1e-6))
        sub = sub.assign(SAMPLE_DAY=times).sort_values("SAMPLE_DAY").reset_index(drop=True)

        for i in range(len(sub)):
            t_start = float(sub.loc[i, "SAMPLE_DAY"])
            if i < len(sub) - 1:
                t_stop = float(sub.loc[i + 1, "SAMPLE_DAY"])
                evt = 0
            else:
                t_stop = float(os_days)
                evt = os_event

            if t_stop <= t_start:
                continue

            row = {
                "PATIENT_ID": pid,
                "CANCER_TYPE": ct,
                "INTERVAL_INDEX": i,
                "T_START": t_start,
                "T_STOP": t_stop,
                "EVENT": evt,
                "OS_DAYS": os_days,
                "OS_EVENT": os_event,
                "SAMPLE_ID": sub.loc[i, "SAMPLE_ID"],
                "SAMPLE_ID_NORM": sub.loc[i, "SAMPLE_ID_NORM"],
                "SAMPLE_DAY": t_start,
                "SAMPLE_TYPE_CODE": sub.loc[i, "SAMPLE_TYPE_CODE"],
                "IS_PRIMARY": bool(sub.loc[i, "IS_PRIMARY"]),
                "MUTATION_BURDEN": float(pd.to_numeric(sub.loc[i, "MUTATION_BURDEN"], errors="coerce") if "MUTATION_BURDEN" in sub.columns else 0.0),
            }
            for cc in clinical_cols:
                row[cc] = prow[cc]
            for oc in omic_cols:
                row[oc] = sub.loc[i, oc] if oc in sub.columns else np.nan

            records.append(row)

    intervals = pd.DataFrame(records)
    out_path = os.path.join(PREP_DIR, "survival_intervals.parquet")
    intervals.to_parquet(out_path, index=False)

    print(f"Saved time-varying survival intervals: {out_path}")
    print(f"Shape: {intervals.shape}")
    print(f"Patients: {intervals['PATIENT_ID'].nunique()} | Events: {intervals['EVENT'].sum()}")
    return intervals


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
class TimeVaryingCoxRidge:
    alpha: float = 1.0
    maxiter: int = 250

    def fit(self, X: np.ndarray, t_start: np.ndarray, t_stop: np.ndarray, event: np.ndarray):
        X = np.asarray(X, dtype=float)
        t_start = np.asarray(t_start, dtype=float)
        t_stop = np.asarray(t_stop, dtype=float)
        event = np.asarray(event, dtype=int)

        ev_idx = np.where(event == 1)[0]
        n_features = X.shape[1]

        def nll_grad(beta: np.ndarray) -> Tuple[float, np.ndarray]:
            eta = np.clip(X @ beta, -40, 40)
            exp_eta = np.exp(eta)

            nll = 0.0
            grad = np.zeros_like(beta)

            for i in ev_idx:
                t = t_stop[i]
                risk = (t_start < t) & (t_stop >= t)
                denom = exp_eta[risk].sum()
                if denom <= 0:
                    continue
                wmean = (exp_eta[risk][:, None] * X[risk]).sum(axis=0) / denom
                nll -= (eta[i] - np.log(denom))
                grad -= (X[i] - wmean)

            nll += 0.5 * self.alpha * np.dot(beta, beta)
            grad += self.alpha * beta
            return float(nll), grad

        beta0 = np.zeros(n_features, dtype=float)
        res = minimize(
            fun=lambda b: nll_grad(b)[0],
            x0=beta0,
            jac=lambda b: nll_grad(b)[1],
            method="L-BFGS-B",
            options={"maxiter": self.maxiter},
        )

        self.coef_ = res.x
        self.success_ = res.success
        self.n_iter_ = res.nit
        if not res.success:
            print(f"Warning: TV-Cox optimizer issue: {res.message}")
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_


def transform_interval_fold(train_df: pd.DataFrame, test_df: pd.DataFrame):
    clinical_cols = [c for c in ["AGE", "SEX", "RACE", "ETHNICITY", "AGE_GROUP", "CANCER_TYPE", "IS_PRIMARY", "SAMPLE_TYPE_CODE", "SAMPLE_DAY"] if c in train_df.columns]
    mutation_cols = [c for c in ["MUTATION_BURDEN"] if c in train_df.columns]
    expr_cols = [c for c in train_df.columns if c.startswith("EXPR_")]
    cna_cols = [c for c in train_df.columns if c.startswith("CNA_")]

    c_train = train_df[clinical_cols].copy() if clinical_cols else pd.DataFrame(index=train_df.index)
    c_test = test_df[clinical_cols].copy() if clinical_cols else pd.DataFrame(index=test_df.index)
    m_train = train_df[mutation_cols].copy() if mutation_cols else pd.DataFrame(index=train_df.index)
    m_test = test_df[mutation_cols].copy() if mutation_cols else pd.DataFrame(index=test_df.index)
    e_train = train_df[expr_cols].copy() if expr_cols else pd.DataFrame(index=train_df.index)
    e_test = test_df[expr_cols].copy() if expr_cols else pd.DataFrame(index=test_df.index)
    n_train = train_df[cna_cols].copy() if cna_cols else pd.DataFrame(index=train_df.index)
    n_test = test_df[cna_cols].copy() if cna_cols else pd.DataFrame(index=test_df.index)

    num_cols = c_train.select_dtypes(include=[np.number, "bool"]).columns.tolist() if not c_train.empty else []
    cat_cols = [c for c in c_train.columns if c not in num_cols] if not c_train.empty else []

    if clinical_cols:
        c_pre = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
                ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
            ],
            remainder="drop",
        )
        Xc_tr = c_pre.fit_transform(c_train)
        Xc_te = c_pre.transform(c_test)
    else:
        Xc_tr = np.empty((len(train_df), 0))
        Xc_te = np.empty((len(test_df), 0))

    if mutation_cols:
        imp_m = SimpleImputer(strategy="median")
        Xm_tr = imp_m.fit_transform(m_train)
        Xm_te = imp_m.transform(m_test)
        sc_m = StandardScaler()
        Xm_tr = sc_m.fit_transform(Xm_tr)
        Xm_te = sc_m.transform(Xm_te)
    else:
        Xm_tr = np.empty((len(train_df), 0))
        Xm_te = np.empty((len(test_df), 0))

    if expr_cols:
        imp_e = SimpleImputer(strategy="constant", fill_value=0.0)
        Xe_tr_raw = imp_e.fit_transform(e_train)
        Xe_te_raw = imp_e.transform(e_test)
        sc_e = StandardScaler()
        Xe_tr_std = sc_e.fit_transform(Xe_tr_raw)
        Xe_te_std = sc_e.transform(Xe_te_raw)
        n_comp_e = min(EXPR_PCA_COMPONENTS, Xe_tr_std.shape[0] - 1, Xe_tr_std.shape[1])
        if n_comp_e > 0:
            pca_e = PCA(n_components=n_comp_e, random_state=RANDOM_STATE)
            Xe_tr = pca_e.fit_transform(Xe_tr_std)
            Xe_te = pca_e.transform(Xe_te_std)
        else:
            Xe_tr = np.empty((len(train_df), 0))
            Xe_te = np.empty((len(test_df), 0))
    else:
        Xe_tr = np.empty((len(train_df), 0))
        Xe_te = np.empty((len(test_df), 0))

    if cna_cols:
        imp_n = SimpleImputer(strategy="constant", fill_value=0.0)
        Xn_tr_raw = imp_n.fit_transform(n_train)
        Xn_te_raw = imp_n.transform(n_test)
        sc_n = StandardScaler()
        Xn_tr_std = sc_n.fit_transform(Xn_tr_raw)
        Xn_te_std = sc_n.transform(Xn_te_raw)
        n_comp_n = min(CNA_PCA_COMPONENTS, Xn_tr_std.shape[0] - 1, Xn_tr_std.shape[1])
        if n_comp_n > 0:
            pca_n = PCA(n_components=n_comp_n, random_state=RANDOM_STATE)
            Xn_tr = pca_n.fit_transform(Xn_tr_std)
            Xn_te = pca_n.transform(Xn_te_std)
        else:
            Xn_tr = np.empty((len(train_df), 0))
            Xn_te = np.empty((len(test_df), 0))
    else:
        Xn_tr = np.empty((len(train_df), 0))
        Xn_te = np.empty((len(test_df), 0))

    return Xc_tr, Xc_te, Xm_tr, Xm_te, Xe_tr, Xe_te, Xn_tr, Xn_te


def run_timevarying_ablation(intervals: pd.DataFrame) -> pd.DataFrame:
    # Patient-level split to avoid leakage across intervals
    p = intervals[["PATIENT_ID", "CANCER_TYPE", "OS_DAYS", "OS_EVENT"]].drop_duplicates(subset=["PATIENT_ID"]).copy()
    p["strata"] = p["CANCER_TYPE"].astype(str) + "_" + p["OS_EVENT"].astype(str)

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

        for fold, (tr_pidx, te_pidx) in enumerate(skf.split(np.zeros(len(p)), p["strata"].values), start=1):
            train_pids = set(p.iloc[tr_pidx]["PATIENT_ID"].tolist())
            test_pids = set(p.iloc[te_pidx]["PATIENT_ID"].tolist())

            tr = intervals[intervals["PATIENT_ID"].isin(train_pids)].copy()
            te = intervals[intervals["PATIENT_ID"].isin(test_pids)].copy()

            Xc_tr, Xc_te, Xm_tr, Xm_te, Xe_tr, Xe_te, Xn_tr, Xn_te = transform_interval_fold(tr, te)

            block_tr = {"clinical": Xc_tr, "mutation": Xm_tr, "expression": Xe_tr, "cna": Xn_tr}
            block_te = {"clinical": Xc_te, "mutation": Xm_te, "expression": Xe_te, "cna": Xn_te}

            Xtr = np.hstack([block_tr[k] for k in includes if block_tr[k].shape[1] > 0])
            Xte = np.hstack([block_te[k] for k in includes if block_te[k].shape[1] > 0])

            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)

            model = TimeVaryingCoxRidge(alpha=RIDGE_ALPHA, maxiter=250)
            model.fit(
                Xtr,
                tr["T_START"].values,
                tr["T_STOP"].values,
                tr["EVENT"].values,
            )

            # interval risk -> patient risk (last observed interval)
            te_risk = model.predict_risk(Xte)
            te_eval = te[["PATIENT_ID", "T_STOP", "OS_DAYS", "OS_EVENT"]].copy()
            te_eval["RISK"] = te_risk
            te_eval = te_eval.sort_values(["PATIENT_ID", "T_STOP"]).drop_duplicates("PATIENT_ID", keep="last")

            cidx = concordance_index_censored(
                te_eval["OS_DAYS"].values,
                te_eval["OS_EVENT"].values,
                te_eval["RISK"].values,
            )
            fold_scores.append(cidx)
            print(f"[{exp_name}] fold {fold}/{N_SPLITS} c-index = {cidx:.4f}  (interval_features={Xtr.shape[1]})")

        rows.append(
            {
                "experiment": exp_name,
                "mean_c_index": float(np.nanmean(fold_scores)),
                "std_c_index": float(np.nanstd(fold_scores)),
                "fold_scores": ", ".join(f"{s:.4f}" for s in fold_scores),
            }
        )

    return pd.DataFrame(rows).sort_values("mean_c_index", ascending=False).reset_index(drop=True)


def main():
    os.makedirs(PREP_DIR, exist_ok=True)

    sample_long = create_sample_omic_long()
    intervals = create_survival_intervals(sample_long)

    results = run_timevarying_ablation(intervals)

    out_csv = os.path.join(PREP_DIR, "ablation_survival_timevarying_results.csv")
    out_json = os.path.join(PREP_DIR, "ablation_survival_timevarying_results.json")
    results.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(results.to_dict(orient="records"), f, indent=2)

    print("\n=== Time-varying Ablation Summary ===")
    print(results.to_string(index=False))
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
