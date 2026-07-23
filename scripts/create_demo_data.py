import pandas as pd
from pathlib import Path

# Paths
cleaned_path = Path("/home/illionar/Projects/ml_research/data/preprocessed_cleaned/patient_multiomic_cleaned.parquet")
output_csv = Path("/home/illionar/Projects/ml_research/demo_input_sample.csv")
output_parquet = Path("/home/illionar/Projects/ml_research/demo_input_sample.parquet")

df = pd.read_parquet(cleaned_path)

# Pick 3 diverse sample patients (e.g. 1 dead early, 1 long survivor, 1 mid)
sample_patients = df.sample(n=3, random_state=42).copy()

sample_patients.to_csv(output_csv, index=False)
sample_patients.to_parquet(output_parquet, index=False)

print(f"Sample created successfully with {len(sample_patients)} patients:")
for idx, row in sample_patients.iterrows():
    print(f" - Patient ID: {row['PATIENT_ID']} | OS_MONTHS: {row['OS_MONTHS']} | OS_EVENT: {row['OS_EVENT']}")
