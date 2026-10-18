
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results" / "duplicate_article_groups.csv"
OUTPUT = ROOT / "results" / "conflicting_duplicate_sample.csv"

df = pd.read_csv(INPUT)

# Each duplicate group has the same article_hash.
df["distinct_summaries"] = pd.to_numeric(
    df["distinct_summaries"], errors="coerce"
)

conflicting = df[df["distinct_summaries"] > 1].copy()

# Sample 10 distinct groups, then include every record from each group.
group_ids = (
    conflicting["article_hash"]
    .drop_duplicates()
    .sample(n=min(10, conflicting["article_hash"].nunique()),
            random_state=42)
)

sample = conflicting[
    conflicting["article_hash"].isin(group_ids)
].copy()

sample = sample.sort_values(
    ["article_hash", "source_row_id"]
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

columns = [c for c in columns if c in sample.columns]
sample[columns].to_csv(OUTPUT, index=False)

print("Conflicting groups available:", conflicting["article_hash"].nunique())
print("Groups sampled:", len(group_ids))
print("Records in sampled groups:", len(sample))
print("Saved:", OUTPUT)

for article_hash, group in sample.groupby("article_hash", sort=False):
    print("\n" + "=" * 80)
    print("Article hash:", article_hash)
    print("Records:", len(group))
    print("Distinct summaries:", group["human_summary"].nunique())

    for _, row in group.iterrows():
        print("\nSource row:", row["source_row_id"])
        print("Headline:", row.get("headline"))
        print("Summary:", str(row.get("human_summary", ""))[:700])
