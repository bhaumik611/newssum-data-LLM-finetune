
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "cleaning_decisions.csv"
OUTPUT = ROOT / "results" / "exclusion_candidates_sample.csv"

df = pd.read_csv(INPUT)

candidates = df[df["proposed_action"] == "exclude_candidate"].copy()

print("Total exclusion candidates:", len(candidates))
print("\nReasons:")
print(candidates["decision_reason"].value_counts().to_string())

# Stratified inspection: sample by reason, rather than only taking
# the first rows in the CSV.
samples = []

for reason, group in candidates.groupby("decision_reason", dropna=False):
    samples.append(
        group.sample(
            n=min(10, len(group)),
            random_state=42
        )
    )

if samples:
    sample_df = pd.concat(samples).drop_duplicates(
        subset=["source_row_id"]
    )
else:
    sample_df = candidates.head(0)

columns = [
    "source_row_id",
    "newspaper_name",
    "headline",
    "article_text",
    "human_summary",
    "decision_reason",
    "flag_url_only_article",
    "flag_summary_refusal_or_error",
    "flag_summary_url_only",
    "flag_summary_instruction",
]

columns = [c for c in columns if c in sample_df.columns]

sample_df[columns].to_csv(OUTPUT, index=False)

print("\nSample records:", len(sample_df))
print("Saved:", OUTPUT)
print("\nSample preview:")
print(sample_df[columns].to_string(index=False, max_colwidth=180))
