import argparse
from pathlib import Path
import pandas as pd


def load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported file type: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export first 5 rows (all columns) to a text file.")
    parser.add_argument("--input", required=True, help="Path to input .parquet or .csv file")
    parser.add_argument("--output", default="preview_first5.txt", help="Output text file path")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    df = load_dataframe(in_path)
    preview = df.head(5).to_string(index=False)

    out_path.write_text(preview + "\n", encoding="utf-8")
    print(f"Wrote {out_path} (first 5 rows, all columns) from {in_path}")


if __name__ == "__main__":
    main()
