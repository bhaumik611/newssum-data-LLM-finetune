
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "raw" / "NewsSumm Dataset.xlsx"

df = pd.read_excel(INPUT)

# Normalize column names
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"\s+", "_", regex=True)
)

print("Checking columns for mixed Python data types...\n")

for col in df.columns:
    types = df[col].dropna().map(
        lambda value: type(value).__name__
    ).value_counts()

    if len(types) > 1:
        print(f"Column: {col}")
        print(types.to_string())
        print("-" * 50)
