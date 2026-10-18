
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results" / "quality_review_all_records.csv"
OUTPUT = ROOT / "results" / "cleaning_decisions.csv"
REPORT = ROOT / "results" / "cleaning_decisions_report.txt"

df = pd.read_csv(INPUT)

# Read flags as booleans. CSV exports may represent them as True/False or 1/0.
flag_cols = [c for c in df.columns if c.startswith("flag_")]

for col in flag_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

df["proposed_action"] = "retain_provisionally"
df["decision_reason"] = "No exclusion rule triggered"

# Rule 1: no usable article body
url_article = df["flag_url_only_article"]
df.loc[url_article, "proposed_action"] = "exclude_candidate"
df.loc[url_article, "decision_reason"] = (
    "Article field contains only a URL; no article body available"
)

# Rule 2: non-summary outputs such as refusals or errors
refusal = df["flag_summary_refusal_or_error"]
df.loc[refusal, "proposed_action"] = "exclude_candidate"
df.loc[refusal, "decision_reason"] = (
    "Summary appears to be a refusal or error response"
)

# Rule 3: no usable summary text
url_summary = df["flag_summary_url_only"]
df.loc[url_summary, "proposed_action"] = "exclude_candidate"
df.loc[url_summary, "decision_reason"] = (
    "Summary field contains only a URL"
)

# Rule 4: instruction contamination requires review, not automatic deletion
instructions = df["flag_summary_instruction"]
df.loc[instructions, "proposed_action"] = "manual_review"
df.loc[instructions, "decision_reason"] = (
    "Summary may contain an instruction or prompt contamination"
)

# Flag overlapping conditions for review.
# Exclusion candidates still need a human confirmation.
for col, reason in [
    ("flag_very_short_article", "Article is unusually short"),
    ("flag_summary_longer_than_article", "Summary exceeds article length"),
    ("flag_extreme_summary_length", "Summary is unusually long"),
    ("flag_missing_headline", "Headline is missing"),
    ("flag_generic_headline", "Headline may be generic or malformed"),
]:
    mask = df[col] & df["proposed_action"].eq("retain_provisionally")
    df.loc[mask, "proposed_action"] = "manual_review"
    df.loc[mask, "decision_reason"] = reason

# Add a blank human decision column.
# Do not modify the original dataset or delete any rows.
df["human_decision"] = ""
df["human_notes"] = ""

df.to_csv(OUTPUT, index=False)

counts = df["proposed_action"].value_counts()

with REPORT.open("w", encoding="utf-8") as f:
    f.write(f"Total records reviewed: {len(df):,}\n\n")
    f.write("Proposed action counts:\n")
    f.write(counts.to_string())
    f.write("\n\nThese are proposed actions, not final exclusions.\n")
    f.write("Review exclusion candidates before creating a training dataset.\n")

print("=" * 55)
print("CLEANING DECISIONS CREATED")
print("=" * 55)
print(f"Total records: {len(df):,}")
print("\nProposed action counts:")
print(counts.to_string())
print(f"\nSaved: {OUTPUT}")
print(f"Saved: {REPORT}")
print("\nNo records were deleted from the prepared dataset.")
