import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path("/home/illionar/Projects/ml_research")
SRC_DIR = BASE_DIR / "data" / "preprocessed"
DST_DIR = BASE_DIR / "data" / "preprocessed_cleaned"

# Columns identified as safe-to-drop with minimal effect from statistical audit.
DROP_MAP = {
    "clinical_all_clean.parquet": [
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
    ],
    "patient_multiomic.parquet": [
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
    ],
    "sample_omic_long.parquet": [
        "DAYS_TO_SPECIMEN_COLLECTION",
    ],
}


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_dir": str(SRC_DIR),
        "output_dir": str(DST_DIR),
        "files": [],
    }

    for file_name, drop_cols in DROP_MAP.items():
        src = SRC_DIR / file_name
        if not src.exists():
            print(f"Skipping missing file: {src}")
            continue

        df = pd.read_parquet(src)
        before_cols = df.shape[1]
        present_drop_cols = [c for c in drop_cols if c in df.columns]
        missing_declared_cols = [c for c in drop_cols if c not in df.columns]

        cleaned = df.drop(columns=present_drop_cols, errors="ignore")
        after_cols = cleaned.shape[1]

        dst = DST_DIR / file_name.replace(".parquet", "_cleaned.parquet")
        cleaned.to_parquet(dst, index=False)

        file_rec = {
            "source_file": file_name,
            "output_file": dst.name,
            "rows": int(cleaned.shape[0]),
            "before_columns": int(before_cols),
            "after_columns": int(after_cols),
            "dropped_count": int(before_cols - after_cols),
            "dropped_columns": present_drop_cols,
            "declared_but_missing_columns": missing_declared_cols,
        }
        manifest["files"].append(file_rec)

        print(
            f"{file_name}: rows={cleaned.shape[0]}, cols {before_cols} -> {after_cols}, "
            f"dropped={before_cols - after_cols}"
        )

    manifest_path = DST_DIR / "cleaning_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
