import argparse
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.config import COLLINEARITY_THRESHOLD, HORIZONS_MONTHS, MAXITER, MC_N_SIMS
from src.utils.io import save_json
from src.data.loader import prepare_dataframe, remove_outliers_iqr, split_three_way, get_feature_sets
from src.features.collinearity import apply_collinearity_filter
from src.features.preprocessing import fit_transform_features, transform_features
from src.training.tuning import run_cv_tuning, select_best_hyperparameters
from src.models.cox_enet import CoxElasticNet
from src.metrics.survival import concordance_index_censored, horizon_test_metrics
from src.models.calibration import fit_horizon_calibrators
from src.training.monte_carlo import monte_carlo_survival_block, survival_probabilities_from_breslow

def main() -> None:
    parser = argparse.ArgumentParser(description="Cox ENet survival pipeline with calibration and Monte Carlo summaries.")
    parser.add_argument("--input", type=Path, default=Path("data/preprocessed_cleaned/patient_multiomic_cleaned.parquet"), help="Input parquet file.")
    parser.add_argument("--outdir", type=Path, default=Path("results/cox_enet_calibrated_mc_outputs_v5"), help="Output directory.")
    parser.add_argument("--no-outlier-filter", action="store_true", help="Keep all rows; default is to keep all rows anyway.")
    parser.add_argument("--apply-outlier-filter", action="store_true", help="Optionally apply IQR filtering on OS_MONTHS and AGE.")
    args = parser.parse_args()

    input_path: Path = args.input
    out_dir: Path = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("model_weights").mkdir(exist_ok=True)

    df = prepare_dataframe(input_path).reset_index(drop=True)

    if args.apply_outlier_filter and not args.no_outlier_filter:
        df, outlier_report = remove_outliers_iqr(df)
        print(f"After outlier filter: {len(df)} rows")
    else:
        outlier_report = {
            "applied": False,
            "reason": "not_used_by_default_to_avoid_target-informed filtering",
            "rows": int(len(df)),
        }

    idx_train, idx_cal, idx_test, split_report = split_three_way(df)

    clinical_cols, expr_cols = get_feature_sets(df)
    print(f"Clinical features: {len(clinical_cols)} | Expression features: {len(expr_cols)}")

    df_train_for_filter = df.iloc[idx_train].copy().reset_index(drop=True)
    clinical_cols_f, expr_cols_f, dropped_df, coll_summary = apply_collinearity_filter(
        df_train_for_filter,
        clinical_cols,
        expr_cols,
        threshold=COLLINEARITY_THRESHOLD,
    )

    dropped_df.to_csv(out_dir / "collinearity_dropped_features.csv", index=False)
    save_json(out_dir / "collinearity_summary.json", coll_summary)

    print(f"After collinearity: clinical {len(clinical_cols)} -> {len(clinical_cols_f)} | expr {len(expr_cols)} -> {len(expr_cols_f)}")

    if not clinical_cols_f and not expr_cols_f:
        raise ValueError("All features removed by collinearity filtering.")

    selected = set(clinical_cols_f) | set(expr_cols_f)
    forbidden_exact = {"OS_MONTHS", "OS_EVENT", "OS_STATUS", "DFS_STATUS", "DFS_EVENT"}
    forbidden_prefixes = ("OS_", "DFS_")
    leaking = sorted(
        c for c in selected if c in forbidden_exact or any(c.startswith(p) for p in forbidden_prefixes)
    )
    if leaking:
        raise ValueError(f"Target-leakage features detected: {leaking}")

    tr = df.iloc[idx_train].copy().reset_index(drop=True)
    cal = df.iloc[idx_cal].copy().reset_index(drop=True)
    te = df.iloc[idx_test].copy().reset_index(drop=True)

    df_train = tr.copy()

    print("Running CV hyperparameter search ...")
    cv_df = run_cv_tuning(df_train, clinical_cols_f, expr_cols_f)
    cv_df.to_csv(out_dir / "hyperparameter_cv_results.csv", index=False)
    best = select_best_hyperparameters(cv_df)
    best_cox_alpha = float(best["alpha"])
    best_l1 = float(best["l1_ratio"])
    print(
        f"Best params: alpha={best_cox_alpha}, l1_ratio={best_l1}, CV c-index={float(best['mean_c_index']):.4f}"
    )

    X_tr, X_cal, clin_pre, expr_pipe, scaler, feat_checks = fit_transform_features(
        tr, cal, clinical_cols_f, expr_cols_f
    )
    X_te = transform_features(te, clinical_cols_f, expr_cols_f, clin_pre, expr_pipe, scaler)

    t_tr = tr["OS_MONTHS"].to_numpy(float)
    e_tr = tr["OS_EVENT"].to_numpy(int)
    t_cal = cal["OS_MONTHS"].to_numpy(float)
    e_cal = cal["OS_EVENT"].to_numpy(int)
    t_te = te["OS_MONTHS"].to_numpy(float)
    e_te = te["OS_EVENT"].to_numpy(int)

    print("Fitting final Cox model ...")
    model = CoxElasticNet(alpha=best_cox_alpha, l1_ratio=best_l1, maxiter=MAXITER)
    model.fit(X_tr, t_tr, e_tr)
    if not bool(getattr(model, "success_", True)):
        raise RuntimeError(f"Cox optimizer did not converge: {getattr(model, 'message_', 'unknown')}")

    r_tr = model.predict_risk(X_tr)
    r_cal = model.predict_risk(X_cal)
    r_te = model.predict_risk(X_te)

    ci_train = float(concordance_index_censored(t_tr, e_tr, r_tr))
    ci_cal = float(concordance_index_censored(t_cal, e_cal, r_cal))
    ci_test = float(concordance_index_censored(t_te, e_te, r_te))
    print(f"C-index: train={ci_train:.4f} | cal={ci_cal:.4f} | test={ci_test:.4f}")

    print("Fitting horizon calibrators ...")
    calibrators = fit_horizon_calibrators(r_cal, t_cal, e_cal, horizons=HORIZONS_MONTHS)
    horizon_metrics, horizon_preds = horizon_test_metrics(
        r_te, t_te, e_te, calibrators, horizons=HORIZONS_MONTHS
    )
    horizon_metrics.to_csv(out_dir / "time_dependent_horizon_metrics.csv", index=False)
    horizon_preds.to_csv(out_dir / "time_dependent_horizon_predictions.csv", index=False)

    print("Running Monte Carlo survival summaries ...")
    mc_te, mc_baseline_meta, mc_summary = monte_carlo_survival_block(
        t_tr,
        e_tr,
        r_tr,
        t_te,
        e_te,
        r_te,
        patient_ids_te=te["PATIENT_ID"].astype(str).to_numpy(),
        n_sims=MC_N_SIMS,
    )
    mc_te.to_csv(out_dir / "monte_carlo_survival_test_predictions.csv", index=False)

    cox_prob_te = survival_probabilities_from_breslow(
        r_te,
        baseline_times=np.asarray(mc_baseline_meta.get("baseline_times", np.array([])), dtype=float),
        baseline_cumhaz=np.asarray(mc_baseline_meta.get("baseline_cumhaz", np.array([])), dtype=float),
        horizons=HORIZONS_MONTHS,
    )

    patient_ids = pd.concat([tr["PATIENT_ID"], cal["PATIENT_ID"], te["PATIENT_ID"]], axis=0).astype(str).values
    pred_df = pd.DataFrame(
        {
            "split": (["train"] * len(tr) + ["calibration"] * len(cal) + ["test"] * len(te)),
            "PATIENT_ID": patient_ids,
            "OS_MONTHS": np.concatenate([t_tr, t_cal, t_te]),
            "OS_EVENT": np.concatenate([e_tr, e_cal, e_te]),
            "risk_score": np.concatenate([r_tr, r_cal, r_te]),
        }
    )

    for h in HORIZONS_MONTHS:
        h = float(h)
        cal_info = calibrators.get(h, {})
        if cal_info.get("status") == "ok":
            iso = cal_info["isotonic"]
            p_event_all = iso.predict(np.concatenate([r_tr, r_cal, r_te]))
            pred_df[f"cal_event_prob_{int(h)}m"] = p_event_all
            pred_df[f"cal_survival_prob_{int(h)}m"] = 1.0 - p_event_all
        else:
            pred_df[f"cal_event_prob_{int(h)}m"] = np.nan
            pred_df[f"cal_survival_prob_{int(h)}m"] = np.nan

    for col in cox_prob_te.columns:
        if col == "risk_score":
            continue
        pred_df[f"test_{col}"] = np.nan
    test_start = len(tr) + len(cal)
    for idx_col in [f"cox_survival_prob_{int(float(h))}m" for h in HORIZONS_MONTHS]:
        pred_df.loc[test_start:test_start + len(te) - 1, f"test_{idx_col}"] = cox_prob_te[idx_col].to_numpy()

    for col in [
        "mc_survival_p10_months",
        "mc_survival_p50_months",
        "mc_survival_p90_months",
        "mc_prob_survive_12_months",
        "mc_prob_survive_24_months",
        "mc_prob_survive_36_months",
        "mc_prob_survive_60_months",
        "mc_rmst_60_months",
    ]:
        pred_df[col] = np.nan
    pred_df.loc[test_start:test_start + len(te) - 1, mc_te.columns.intersection(pred_df.columns)] = mc_te[
        mc_te.columns.intersection(pred_df.columns)
    ].to_numpy()

    pred_df.to_csv(out_dir / "main_predictions.csv", index=False)
    pred_df.to_csv(out_dir / "tuned_model_predictions.csv", index=False)

    coef_df = pd.DataFrame(
        {"coef_index": np.arange(len(model.coef_)), "coef_value": model.coef_}
    ).sort_values("coef_value", key=np.abs, ascending=False)
    coef_df.to_csv(out_dir / "coefficient_exports.csv", index=False)
    coef_df.to_csv(out_dir / "tuned_model_coefficients.csv", index=False)

    mc_widths = (mc_te["mc_survival_p90_months"] - mc_te["mc_survival_p10_months"]).to_numpy(dtype=float)
    mc_widths = mc_widths[np.isfinite(mc_widths)]
    mc_mean_width = float(np.mean(mc_widths)) if len(mc_widths) else float("nan")
    mc_median_width = float(np.median(mc_widths)) if len(mc_widths) else float("nan")
    mc_p90_width = float(np.percentile(mc_widths, 90)) if len(mc_widths) else float("nan")

    metrics: Dict[str, Any] = {
        "input_file": str(input_path),
        "best_cox_params": {"alpha": best_cox_alpha, "l1_ratio": best_l1},
        "cv_best_mean_c_index": float(best["mean_c_index"]),
        "cv_best_std_c_index": float(best["std_c_index"]),
        "cv_best_valid_folds": int(best["n_valid_folds"]),
        "n_total_rows": int(len(df)),
        "n_train": int(len(tr)),
        "n_calibration": int(len(cal)),
        "n_test": int(len(te)),
        "events_train": int(e_tr.sum()),
        "events_calibration": int(e_cal.sum()),
        "events_test": int(e_te.sum()),
        "cox_optimizer_success": bool(getattr(model, "success_", True)),
        "cox_optimizer_message": str(getattr(model, "message_", "unknown")),
        "cox_optimizer_iterations": int(getattr(model, "n_iter_", -1)),
        "c_index_train": ci_train,
        "c_index_calibration": ci_cal,
        "c_index_test": ci_test,
        "collinearity_filter": {
            "threshold_abs_pearson": COLLINEARITY_THRESHOLD,
            "clinical_before": int(len(clinical_cols)),
            "clinical_after": int(len(clinical_cols_f)),
            "expr_before": int(len(expr_cols)),
            "expr_after": int(len(expr_cols_f)),
        },
        "horizon_metrics": horizon_metrics.to_dict(orient="records"),
        "monte_carlo": {
            **mc_summary,
            "mean_interval_width_months": mc_mean_width,
            "median_interval_width_months": mc_median_width,
            "p90_interval_width_months": mc_p90_width,
        },
    }

    save_json(out_dir / "metrics.json", metrics)
    save_json(out_dir / "tuned_model_metrics.json", metrics)

    consistency: Dict[str, Any] = {
        "input_file": str(input_path),
        "rows_before_outlier_filter": int(len(df)),
        "outlier_report": outlier_report,
        "split_report": split_report,
        "feature_checks": feat_checks,
        "feature_counts": {
            "clinical_before_collinearity": int(len(clinical_cols)),
            "expr_before_collinearity": int(len(expr_cols)),
            "clinical_after_collinearity": int(len(clinical_cols_f)),
            "expr_after_collinearity": int(len(expr_cols_f)),
        },
        "collinearity_summary": coll_summary,
        "leakage_check_passed": True,
        "monte_carlo_baseline_meta": {
            k: v for k, v in mc_baseline_meta.items() if k not in {"baseline_times", "baseline_cumhaz"}
        },
    }
    save_json(out_dir / "consistency_checks.json", consistency)
    save_json(out_dir / "audit.json", consistency)

    artifact: Dict[str, Any] = {
        "clinical_cols_before_collinearity": clinical_cols,
        "expr_cols_before_collinearity": expr_cols,
        "clinical_cols_after_collinearity": clinical_cols_f,
        "expr_cols_after_collinearity": expr_cols_f,
        "clin_pre": clin_pre,
        "expr_pipe": expr_pipe,
        "scaler": scaler,
        "cox_model": model,
        "best_cox_alpha": best_cox_alpha,
        "best_cox_l1_ratio": best_l1,
        "horizon_calibrators": calibrators,
        "horizons_months": HORIZONS_MONTHS,
        "mc_summary": mc_summary,
        "mc_baseline_meta": {
            k: v for k, v in mc_baseline_meta.items() if k not in {"baseline_times", "baseline_cumhaz"}
        },
    }
    with open(Path("model_weights") / "final_locked_model.pkl", "wb") as f:
        pickle.dump(artifact, f)

    readme = [
        "# Cox ENet + calibrated horizon risk estimation + Monte Carlo",
        "",
        "## What this pipeline reports",
        "- risk score from a fitted Cox Elastic-Net model",
        "- horizon-wise calibrated event and survival probabilities at 12 / 24 / 36 / 60 months",
        "- C-index on train / calibration / test",
        "- horizon AUROC and Brier score on known-label test rows",
        "- Monte Carlo survival summaries from the fitted Cox model and Breslow baseline hazard",
        "- bounded RMST at 60 months",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print("\nDone.")
    print(f"Results -> {out_dir.resolve()}")

if __name__ == "__main__":
    main()
