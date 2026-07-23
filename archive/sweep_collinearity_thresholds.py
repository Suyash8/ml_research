import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

import tune_train_cox_enet_conformal_td_outlier_collinearity as run_mod


BASE = Path("/home/illionar/Projects/ml_research")
MODEL_OUT_BASE = BASE / "data" / "model_outputs"
SWEEP_OUT_DIR = MODEL_OUT_BASE / "collinearity_threshold_sweep"

DEFAULT_THRESHOLDS = [1.0, 0.95, 0.90, 0.85, 0.75]


def threshold_label(threshold: float) -> str:
    return f"{int(round(threshold * 100)):03d}"


def collect_metrics(out_dir: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {}

    metrics_path = out_dir / "tuned_model_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        month = metrics.get("conformal_month_interval", {})
        row.update(
            {
                "cv_best_mean_c_index": metrics.get("cv_best_mean_c_index"),
                "c_index_train": metrics.get("c_index_train"),
                "c_index_calibration": metrics.get("c_index_calibration"),
                "c_index_test": metrics.get("c_index_test"),
                "coverage_test_events_only": month.get("coverage_test_events_only"),
                "mean_interval_width_months_test": month.get("mean_interval_width_months_test"),
                "n_train": metrics.get("n_train"),
                "n_calibration": metrics.get("n_calibration"),
                "n_test": metrics.get("n_test"),
                "events_train": metrics.get("events_train"),
                "events_calibration": metrics.get("events_calibration"),
                "events_test": metrics.get("events_test"),
            }
        )

    col_summary_path = out_dir / "collinearity_summary.json"
    if col_summary_path.exists():
        col = json.loads(col_summary_path.read_text(encoding="utf-8"))
        counts = col.get("feature_counts", {})
        expr = col.get("expression", {})
        clin = col.get("clinical_numeric", {})
        row.update(
            {
                "expr_before": counts.get("expr_before"),
                "expr_after": counts.get("expr_after"),
                "expr_dropped": expr.get("dropped_count"),
                "clinical_before": counts.get("clinical_before"),
                "clinical_after": counts.get("clinical_after"),
                "clinical_dropped": clin.get("dropped_count"),
                "collinearity_pairs_above_threshold": expr.get("method_details", {}).get("n_pairs_above_threshold"),
            }
        )

    return row


def render_markdown_table(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in df.itertuples(index=False, name=None):
        vals: List[str] = []
        for v in row:
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def run_one_threshold(threshold: float, rerun_existing: bool) -> Dict[str, Any]:
    label = threshold_label(threshold)
    out_dir = MODEL_OUT_BASE / f"cox_enet_conformal_tuned_td_outlier_collinearity_{label}"

    metrics_path = out_dir / "tuned_model_metrics.json"
    col_summary_path = out_dir / "collinearity_summary.json"

    row: Dict[str, Any] = {
        "threshold": float(threshold),
        "threshold_label": label,
        "out_dir": str(out_dir),
        "status": "pending",
        "error": "",
    }

    should_run = rerun_existing or not (metrics_path.exists() and col_summary_path.exists())
    if should_run:
        run_mod.COLLINEARITY_THRESHOLD = float(threshold)
        run_mod.OUT_DIR = out_dir

        try:
            run_mod.main()
            row["status"] = "completed"
        except Exception:
            row["status"] = "failed"
            row["error"] = traceback.format_exc()
    else:
        row["status"] = "skipped_existing"

    row.update(collect_metrics(out_dir))
    return row


def build_markdown_report(df: pd.DataFrame, out_md: Path) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    completed = df[df["status"].isin(["completed", "skipped_existing"])].copy()
    completed = completed.dropna(subset=["c_index_test"])
    if completed.empty:
        lines = [
            "# Collinearity Threshold Sweep",
            "",
            f"Generated: {now}",
            "",
            "No successful runs with metrics were available.",
        ]
        out_md.write_text("\n".join(lines), encoding="utf-8")
        return

    no_drop = completed.loc[completed["threshold"] == completed["threshold"].max()]
    if not no_drop.empty:
        ref = no_drop.iloc[0]
        ref_test = ref["c_index_test"]
        ref_cov = ref["coverage_test_events_only"]
        ref_width = ref["mean_interval_width_months_test"]

        completed["delta_test_c_index_vs_no_drop"] = completed["c_index_test"] - ref_test
        completed["delta_coverage_vs_no_drop"] = completed["coverage_test_events_only"] - ref_cov
        completed["delta_width_vs_no_drop"] = completed["mean_interval_width_months_test"] - ref_width
    else:
        completed["delta_test_c_index_vs_no_drop"] = pd.NA
        completed["delta_coverage_vs_no_drop"] = pd.NA
        completed["delta_width_vs_no_drop"] = pd.NA

    best_cidx = completed.sort_values("c_index_test", ascending=False).iloc[0]
    best_width = completed.sort_values("mean_interval_width_months_test", ascending=True).iloc[0]

    display_cols = [
        "threshold",
        "expr_after",
        "expr_dropped",
        "c_index_test",
        "coverage_test_events_only",
        "mean_interval_width_months_test",
        "delta_test_c_index_vs_no_drop",
        "delta_coverage_vs_no_drop",
        "delta_width_vs_no_drop",
        "status",
    ]

    table_df = completed[display_cols].sort_values("threshold", ascending=False).copy()

    for c in [
        "threshold",
        "c_index_test",
        "coverage_test_events_only",
        "mean_interval_width_months_test",
        "delta_test_c_index_vs_no_drop",
        "delta_coverage_vs_no_drop",
        "delta_width_vs_no_drop",
    ]:
        table_df[c] = pd.to_numeric(table_df[c], errors="coerce").round(4)

    lines = [
        "# Collinearity Threshold Sweep",
        "",
        f"Generated: {now}",
        "",
        "## Run Comparison",
        render_markdown_table(table_df),
        "",
        "## Key Selections",
        f"- Best test c-index: threshold={best_cidx['threshold']:.2f}, c-index={best_cidx['c_index_test']:.4f}",
        f"- Tightest mean interval width: threshold={best_width['threshold']:.2f}, width={best_width['mean_interval_width_months_test']:.4f} months",
        "",
        "## Notes",
        "- Threshold 1.00 acts as no-drop baseline (since dropping is applied only when correlation > threshold).",
        "- Expression drop counts are based on train split only and then applied consistently to calibration/test.",
    ]

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run collinearity threshold sweep for Cox ENet pipeline.")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Absolute Pearson correlation thresholds to evaluate.",
    )
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Re-run thresholds even if output files already exist.",
    )
    args = parser.parse_args()

    SWEEP_OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for threshold in args.thresholds:
        rows.append(run_one_threshold(float(threshold), rerun_existing=bool(args.rerun_existing)))

    df = pd.DataFrame(rows)
    df = df.sort_values("threshold", ascending=False).reset_index(drop=True)

    out_csv = SWEEP_OUT_DIR / "threshold_sweep_summary.csv"
    df.to_csv(out_csv, index=False)

    out_md = SWEEP_OUT_DIR / "threshold_sweep_summary.md"
    build_markdown_report(df, out_md)

    print("Done.")
    print(f"Summary CSV: {out_csv}")
    print(f"Summary MD:  {out_md}")


if __name__ == "__main__":
    main()
