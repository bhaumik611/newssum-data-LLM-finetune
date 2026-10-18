
from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/raw/NewsSumm Dataset.xlsx")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT_FILE.resolve()}")

print("Loading dataset...")
df = pd.read_excel(INPUT_FILE)

# Normalize column names, including the trailing newline.
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"\s+", "_", regex=True)
)

print("\n=== BASIC INFORMATION ===")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("Memory usage (MB):",
      round(df.memory_usage(deep=True).sum() / 1024**2, 2))

required = {
    "newspaper_name", "published_date", "headline",
    "article_text", "human_summary", "news_category"
}
missing_columns = required - set(df.columns)

if missing_columns:
    raise ValueError(f"Unexpected schema. Missing: {missing_columns}")

# Check missing and blank values.
print("\n=== MISSING VALUES ===")
for col in required:
    values = df[col]
    blank = (
        values.astype("string").str.strip().eq("").fillna(False)
        if values.dtype == "object" or
           isinstance(values.dtype, pd.StringDtype)
        else pd.Series(False, index=df.index)
    )
    print(
        f"{col:20} nulls={values.isna().sum():7} "
        f"blanks={blank.sum():7}"
    )

# Parse dates without silently discarding invalid values.
df["published_date"] = pd.to_datetime(
    df["published_date"], errors="coerce"
)

print("\n=== DATE RANGE ===")
print("Valid dates:", df["published_date"].notna().sum())
print("Invalid dates:", df["published_date"].isna().sum())
print("Earliest:", df["published_date"].min())
print("Latest:", df["published_date"].max())

# Detect exact duplicates.
print("\n=== DUPLICATES ===")
print("Duplicate complete rows:", df.duplicated().sum())
print(
    "Duplicate normalized headlines:",
    df["headline"].fillna("").str.strip().str.lower().duplicated().sum()
)
print(
    "Duplicate article texts:",
    df["article_text"].fillna("").str.strip().str.lower().duplicated().sum()
)

# Text-length statistics; lengths are in characters, not tokens.
for col in ["headline", "article_text", "human_summary"]:
    lengths = df[col].fillna("").astype(str).str.len()
    print(f"\n=== {col.upper()} LENGTHS (CHARACTERS) ===")
    print(lengths.describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

print("\n=== TOP CATEGORIES ===")
print(df["news_category"].value_counts(dropna=False).head(20))

print("\n=== TOP NEWSPAPERS ===")
print(df["newspaper_name"].value_counts(dropna=False).head(20))

# Inspect possible headline/content mismatches without printing full articles.
print("\n=== SAMPLE METADATA ===")
for _, row in df.sample(min(10, len(df)), random_state=42).iterrows():
    print({
        "headline": str(row["headline"])[:120],
        "newspaper": row["newspaper_name"],
        "date": str(row["published_date"]),
        "category": row["news_category"],
        "article_chars": len(str(row["article_text"])),
        "summary_chars": len(str(row["human_summary"])),
    })

# Save a compact machine-readable report.
report = {
    "rows": int(len(df)),
    "columns": df.columns.tolist(),
    "missing_values": {
        col: int(df[col].isna().sum()) for col in df.columns
    },
    "duplicate_complete_rows": int(df.duplicated().sum()),
    "duplicate_headlines": int(
        df["headline"].fillna("").str.strip().str.lower().duplicated().sum()
    ),
    "duplicate_article_texts": int(
        df["article_text"].fillna("").str.strip().str.lower().duplicated().sum()
    ),
    "earliest_date": str(df["published_date"].min()),
    "latest_date": str(df["published_date"].max()),
    "category_counts": {
        str(k): int(v)
        for k, v in df["news_category"].value_counts(dropna=False).items()
    },
}

report_path = OUTPUT_DIR / "dataset_audit.json"
report_path.write_text(
    __import__("json").dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\nAudit report saved to:", report_path)
