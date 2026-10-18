
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PREPARED = ROOT / "data" / "processed" / "newssumm_clean_candidate.parquet"
DECISIONS = ROOT / "results" / "cleaning_decisions.csv"

OUTPUT = ROOT / "data" / "processed" / "newssumm_initial_training.parquet"
EXCLUDED = ROOT / "results" / "initial_exclusions.csv"
REPORT = ROOT / "results" / "initial_training_dataset_report.txt"

prepared = pd.read_parquet(PREPARED)
decisions = pd.read_csv(DECISIONS)

# Verify that source-row identifiers match before applying decisions.
if "source_row_id" not in prepared or "source_row_id" not in decisions:
    raise ValueError("source_row_id is required in both files")

if prepared["source_row_id"].duplicated().any():
    raise ValueError("Prepared dataset has duplicate source_row_id values")

if decisions["source_row_id"].duplicated().any():
    raise ValueError("Decision file has duplicate source_row_id values")

if set(prepared["source_row_id"]) != set(decisions["source_row_id"]):
    raise ValueError("Prepared dataset and decision file have different row IDs")

# Only exclude records with a reviewed, explicit candidate reason.
# This is an initial dataset, not a claim that all remaining data is clean.
exclude_ids = set(
    decisions.loc[
        decisions["proposed_action"].eq("exclude_candidate"),
        "source_row_id",
    ]
)

excluded = prepared[
    prepared["source_row_id"].isin(exclude_ids)
].copy()

initial = prepared[
    ~prepared["source_row_id"].isin(exclude_ids)
].copy()

# Basic safeguards
for col in ["article_text", "human_summary"]:
    if initial[col].isna().any():
        raise ValueError(f"Missing values remain in {col}")

if len(initial) + len(excluded) != len(prepared):
    raise ValueError("Row-count reconciliation failed")

initial.to_parquet(OUTPUT, index=False)

# Preserve decisions and reasons for auditability.
excluded_log = decisions[
    decisions["source_row_id"].isin(exclude_ids)
].copy()
excluded_log.to_csv(EXCLUDED, index=False)

with REPORT.open("w", encoding="utf-8") as f:
    f.write("INITIAL TRAINING DATASET REPORT\n")
    f.write("=" * 40 + "\n")
    f.write(f"Prepared records: {len(prepared):,}\n")
    f.write(f"Excluded candidates: {len(excluded):,}\n")
    f.write(f"Initial remaining records: {len(initial):,}\n")
    f.write("\n")
    f.write("This dataset is provisional. It has not undergone complete\n")
    f.write("semantic alignment or factuality validation.\n")

print("Initial dataset created successfully.")
print(f"Prepared records: {len(prepared):,}")
print(f"Excluded candidates: {len(excluded):,}")
print(f"Remaining records: {len(initial):,}")
print(f"\nDataset: {OUTPUT}")
print(f"Exclusion log: {EXCLUDED}")
print(f"Report: {REPORT}")
