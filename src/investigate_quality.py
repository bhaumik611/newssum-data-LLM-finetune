
from pathlib import Path
import pandas as pd

INPUT = Path("data/raw/NewsSumm Dataset.xlsx")
OUT = Path("results")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(INPUT)
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"\s+", "_", regex=True)
)

# Preserve the original strings before any date conversion.
df["published_date_original"] = df["published_date"].astype("string")
df["date_parsed"] = pd.to_datetime(
    df["published_date"], errors="coerce"
)

print("\n=== DATE ANOMALIES ===")

# This is a diagnostic range, not an automatic deletion rule.
date_outliers = df[
    df["date_parsed"].isna()
    | (df["date_parsed"] < "2000-01-01")
    | (df["date_parsed"] > "2026-12-31")
]

print("Date anomalies:", len(date_outliers))
print(
    date_outliers[
        ["newspaper_name", "published_date_original", "headline"]
    ].head(30).to_string(index=False)
)

date_outliers[
    ["newspaper_name", "published_date_original", "headline"]
].to_csv(OUT / "date_anomalies.csv", index=False)

print("\n=== EXACT DUPLICATES ===")
print("Duplicate complete rows:", df.duplicated().sum())

# Normalize text for diagnostic matching only.
for col in ["headline", "article_text", "human_summary"]:
    df[f"{col}_norm"] = (
        df[col].astype("string")
        .str.normalize("NFKC")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.casefold()
    )

for col in ["headline_norm", "article_text_norm"]:
    counts = df[col].value_counts(dropna=True)
    repeated = counts[counts > 1]

    print(f"\n{col}")
    print("Distinct repeated values:", len(repeated))
    print("Rows belonging to repeated groups:", int(repeated.sum()))
    print("Occurrences after the first:", int((repeated - 1).sum()))

print("\n=== REPEATED ARTICLE TEXTS WITH DIFFERENT SUMMARIES ===")

text_groups = df[
    df["article_text_norm"].notna()
    & (df["article_text_norm"] != "")
].groupby("article_text_norm", sort=False)

examples = []

for text, group in text_groups:
    if len(group) > 1:
        summaries = group["human_summary_norm"].dropna().unique()
        if len(summaries) > 1:
            examples.append({
                "article_text_prefix": text[:200],
                "row_count": len(group),
                "different_summaries": len(summaries),
                "headlines": " | ".join(
                    group["headline"].dropna().astype(str).head(3)
                ),
            })

        if len(examples) >= 100:
            break

conflicts = pd.DataFrame(examples)
conflicts.to_csv(OUT / "duplicate_summary_conflicts.csv", index=False)

print("Conflicting duplicate-text examples inspected:", len(conflicts))
print(conflicts.head(10).to_string(index=False))

print("\n=== CATEGORY NORMALIZATION CANDIDATE ===")

df["category_normalized"] = (
    df["news_category"].astype("string")
    .str.normalize("NFKC")
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
    .str.casefold()
)

print("Original distinct categories:", df["news_category"].nunique())
print("After case/whitespace normalization:",
      df["category_normalized"].nunique())

print("\nMost frequent normalized categories:")
print(df["category_normalized"].value_counts().head(30))

print("\n=== SUMMARY QUALITY ===")

summary = df["human_summary"].fillna("").astype(str).str.strip()
article = df["article_text"].fillna("").astype(str).str.strip()

print("Empty summaries:", int(summary.eq("").sum()))
print("Empty articles:", int(article.eq("").sum()))
print("Summaries longer than articles:",
      int((summary.str.len() > article.str.len()).sum()))

print("\nReports saved in:", OUT.resolve())
