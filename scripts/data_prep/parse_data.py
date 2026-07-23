# view_parquet_preview.py
import argparse
from pathlib import Path
import pandas as pd

DEFAULTS = {
    "clinical": "data/preprocessed/clinical_all_clean.parquet",
    "multiomic": "data/preprocessed/patient_multiomic.parquet",
    "sample_long": "data/preprocessed/sample_omic_long.parquet",
    "intervals": "data/preprocessed/survival_intervals.parquet",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None, help="Path to parquet file")
    parser.add_argument("--preset", type=str, default="clinical", choices=DEFAULTS.keys())
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int, default=12)
    args = parser.parse_args()

    path = Path(args.path) if args.path else Path(DEFAULTS[args.preset])
    print("Using:", path)
    df = pd.read_parquet(path)

    # pick sensible columns
    cols = [c for c in df.columns]

    # add a couple of omics columns if present
    expr = [c for c in df.columns if c.startswith("EXPR_")]
    cna = [c for c in df.columns if c.startswith("CNA_")]
    cols += expr[:2] + cna[:2]

    if not cols:
        cols = df.columns[:args.cols].tolist()
    else:
        cols = cols[:args.cols]

    print(df[cols].head(args.rows).to_string(index=False))

if __name__ == "__main__":
    main()