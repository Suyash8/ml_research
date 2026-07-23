import json
from pathlib import Path

import pandas as pd

from build_longitudinal_and_ablate import run_timevarying_ablation


BASE = Path("/home/illionar/Projects/ml_research")
PREP = BASE / "data" / "preprocessed"
PREP_CLEAN = BASE / "data" / "preprocessed_cleaned"

# Same statistically low-value columns previously removed from clinical/patient tables.
DROP_COLS = {
    "HISTORY_LGG_DX_OF_BRAIN_TISSUE",
    "PERFORMANCE_STATUS_TIMING",
    "PHARMACEUTICAL_TX_ADJUVANT",
    "FAMILY_HISTORY_OF_CANCER",
    "HISTORY_HEPATO_CARCINOMA_RISK_FACTORS",
    "DEFINITIVE_SURGICAL_PROCEDURE",
    "PRIMARY_MELANOMA_KNOWN_DX",
    "PRIMARY_MULTIPLE_AT_DX",
    "SUBMITTED_TUMOR_DX_DAYS_TO",
    "TUMOR_SITE",
    "PRIMARY_MELANOMA_SKIN_TYPE",
    "DAYS_TO_SPECIMEN_COLLECTION",
}


def main() -> None:
    PREP_CLEAN.mkdir(parents=True, exist_ok=True)

    intervals_path = PREP / "survival_intervals.parquet"
    if not intervals_path.exists():
        raise FileNotFoundError(f"Missing: {intervals_path}")

    intervals = pd.read_parquet(intervals_path)

    # Apply same drop policy if any of those columns exist in interval table.
    present_drop = [c for c in intervals.columns if c in DROP_COLS]
    intervals_clean = intervals.drop(columns=present_drop, errors="ignore")

    print(f"Loaded intervals: {intervals.shape}")
    print(f"Dropped columns present in intervals: {len(present_drop)}")
    if present_drop:
        print("Dropped:", present_drop)
    print(f"Intervals used for cleaned rerun: {intervals_clean.shape}")

    results = run_timevarying_ablation(intervals_clean)

    out_csv = PREP_CLEAN / "ablation_survival_timevarying_results_cleaned.csv"
    out_json = PREP_CLEAN / "ablation_survival_timevarying_results_cleaned.json"
    results.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(results.to_dict(orient="records"), indent=2), encoding="utf-8")

    print("\n=== Cleaned Time-varying Ablation Summary ===")
    print(results.to_string(index=False))
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_json}")

    original_csv = PREP / "ablation_survival_timevarying_results.csv"
    if original_csv.exists():
        old = pd.read_csv(original_csv)
        merged = old.merge(
            results,
            on="experiment",
            suffixes=("_old", "_cleaned"),
            how="inner",
        )
        merged["delta_mean_c_index"] = merged["mean_c_index_cleaned"] - merged["mean_c_index_old"]
        cmp_csv = PREP_CLEAN / "ablation_survival_timevarying_results_comparison.csv"
        merged.sort_values("mean_c_index_cleaned", ascending=False).to_csv(cmp_csv, index=False)
        print(f"Saved comparison: {cmp_csv}")
        print("\nMean c-index deltas (cleaned - old):")
        print(
            merged[["experiment", "mean_c_index_old", "mean_c_index_cleaned", "delta_mean_c_index"]]
            .sort_values("delta_mean_c_index", ascending=False)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
