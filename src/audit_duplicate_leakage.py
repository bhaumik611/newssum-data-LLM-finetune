
from pathlib import Path
import hashlib
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "processed" / "newssumm_initial_training.parquet"
GROUPS_OUTPUT = ROOT / "results" / "duplicate_article_groups.csv"
REPORT_OUTPUT = ROOT / "results" / "duplicate_article_audit.txt"

df = pd.read_parquet(INPUT)

def normalize_text(value):
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def text_hash(value):
    normalized = normalize_text(value)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

# Hash normalized article text to identify exact matches after
# case and whitespace normalization.
df["article_hash"] = df["article_text"].map(text_hash)

# Empty articles should never be grouped as duplicates.
empty_mask = df["article_text"].map(normalize_text).eq("")
if empty_mask.any():
    raise ValueError(f"Found {empty_mask.sum()} empty article texts")

group_sizes = df.groupby("article_hash")["source_row_id"].transform("size")
duplicate_rows = df[group_sizes > 1].copy()

group_counts = (
    duplicate_rows.groupby("article_hash")
    .agg(
        record_count=("source_row_id", "size"),
        distinct_summaries=(
            "human_summary",
            lambda x: x.map(normalize_text).nunique(),
        ),
        distinct_headlines=(
            "headline",
            lambda x: x.map(normalize_text).nunique(),
        ),
    )
    .reset_index()
)

duplicate_rows = duplicate_rows.merge(
    group_counts,
    on="article_hash",
    how="left",
    validate="many_to_one",
)

duplicate_rows = duplicate_rows.sort_values(
    ["record_count", "article_hash"],
    ascending=[False, True],
)

columns = [
    "source_row_id",
    "newspaper_name",
    "published_date_original",
    "headline",
    "article_text",
    "human_summary",
    "article_hash",
    "record_count",
    "distinct_summaries",
    "distinct_headlines",
]
columns = [c for c in columns if c in duplicate_rows.columns]

duplicate_rows[columns].to_csv(GROUPS_OUTPUT, index=False)

duplicate_groups = len(group_counts)
rows_in_groups = len(duplicate_rows)
extra_rows = int((group_counts["record_count"] - 1).sum())
groups_with_conflicting_summaries = int(
    (group_counts["distinct_summaries"] > 1).sum()
)

with REPORT_OUTPUT.open("w", encoding="utf-8") as f:
    f.write("DUPLICATE ARTICLE AUDIT\n")
    f.write("=" * 40 + "\n")
    f.write(f"Dataset records: {len(df):,}\n")
    f.write(f"Unique normalized article texts: {df['article_hash'].nunique():,}\n")
    f.write(f"Duplicate article groups: {duplicate_groups:,}\n")
    f.write(f"Records belonging to duplicate groups: {rows_in_groups:,}\n")
    f.write(f"Extra occurrences beyond one per group: {extra_rows:,}\n")
    f.write(
        "Groups with conflicting normalized summaries: "
        f"{groups_with_conflicting_summaries:,}\n"
    )
    f.write("\nNo records were deleted or changed.\n")

print("=" * 55)
print("DUPLICATE ARTICLE AUDIT")
print("=" * 55)
print(f"Dataset records: {len(df):,}")
print(f"Unique normalized article texts: {df['article_hash'].nunique():,}")
print(f"Duplicate article groups: {duplicate_groups:,}")
print(f"Records in duplicate groups: {rows_in_groups:,}")
print(f"Extra occurrences: {extra_rows:,}")
print(f"Groups with conflicting summaries: {groups_with_conflicting_summaries:,}")
print(f"\nSaved: {GROUPS_OUTPUT}")
print(f"Saved: {REPORT_OUTPUT}")
