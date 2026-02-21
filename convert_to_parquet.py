import os
import pandas as pd

PREP = "/home/illionar/Projects/ml_research/data/preprocessed"

csv_files = [f for f in os.listdir(PREP) if f.endswith(".csv")]
print(f"Converting {len(csv_files)} CSV files to Parquet (snappy compression)...\n")

for fname in sorted(csv_files):
    csv_path = os.path.join(PREP, fname)
    pq_path  = os.path.join(PREP, fname.replace(".csv", ".parquet"))

    csv_mb = os.path.getsize(csv_path) / 1024**2
    print(f"  Reading  {fname} ...", flush=True)
    df = pd.read_csv(csv_path, low_memory=False)
    # Cast any object columns with mixed types to str so pyarrow can infer a clean schema
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).where(df[col].notna(), other=None)
    df.to_parquet(pq_path, index=False, compression="snappy", engine="pyarrow")
    pq_mb = os.path.getsize(pq_path) / 1024**2

    os.remove(csv_path)
    ratio = (1 - pq_mb / csv_mb) * 100
    print(f"  ✓  {fname:45s}  {csv_mb:7.1f} MB  →  {pq_mb:5.1f} MB  ({ratio:.0f}% smaller)")

print("\nDone. Final directory listing:")
for f in sorted(os.listdir(PREP)):
    size_mb = os.path.getsize(os.path.join(PREP, f)) / 1024**2
    print(f"  {f:55s}  {size_mb:.1f} MB")
