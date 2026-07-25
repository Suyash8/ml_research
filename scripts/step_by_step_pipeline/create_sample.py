import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
clean_path = ROOT_DIR / "data" / "preprocessed_cleaned" / "patient_multiomic_cleaned.parquet"
output_csv = Path(__file__).resolve().parent / "00_sample_patient_data.csv"

if clean_path.exists():
    df = pd.read_parquet(clean_path)
    sample_df = df.sample(n=5, random_state=42).copy()
    sample_df.to_csv(output_csv, index=False)
    print(f"✅ Saved 5 sample patient records to {output_csv}")
else:
    print(f"❌ Clean dataset not found at {clean_path}")
