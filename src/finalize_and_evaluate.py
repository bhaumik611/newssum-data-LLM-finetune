
from pathlib import Path
import hashlib
import json
import re

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/newssumm_initial_training.parquet"
OUT = ROOT / "data/processed/splits"
RESULTS = ROOT / "results"

OUT.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

SEED = 42

# 1. Load the provisional dataset
df = pd.read_parquet(INPUT)
required = {"source_row_id", "article_text", "human_summary"}
if not required.issubset(df.columns):
    raise ValueError(f"Missing columns: {required - set(df.columns)}")

df = df.dropna(subset=["article_text", "human_summary"]).copy()
df["article_text"] = df["article_text"].astype(str)
df["human_summary"] = df["human_summary"].astype(str)

def normalize(text):
    return re.sub(r"\s+", " ", str(text).lower()).strip()

def article_hash(text):
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()

df["duplicate_group_id"] = df["article_text"].map(article_hash)

# 2. Group-aware 80/10/10 split
# All normalized-identical article texts stay in one partition.
split1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
train_idx, temp_idx = next(
    split1.split(df, groups=df["duplicate_group_id"])
)

train = df.iloc[train_idx].copy()
temp = df.iloc[temp_idx].copy()

split2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
val_idx, test_idx = next(
    split2.split(temp, groups=temp["duplicate_group_id"])
)

val = temp.iloc[val_idx].copy()
test = temp.iloc[test_idx].copy()

# Verify no exact normalized article group crosses partitions.
train_groups = set(train["duplicate_group_id"])
val_groups = set(val["duplicate_group_id"])
test_groups = set(test["duplicate_group_id"])

assert not train_groups & val_groups
assert not train_groups & test_groups
assert not val_groups & test_groups

for name, part in [
    ("train", train), ("validation", val), ("test", test)
]:
    part.to_parquet(OUT / f"{name}.parquet", index=False)

split_report = {
    "seed": SEED,
    "split_method": "GroupShuffleSplit by normalized article text hash",
    "train_rows": len(train),
    "validation_rows": len(val),
    "test_rows": len(test),
    "train_groups": len(train_groups),
    "validation_groups": len(val_groups),
    "test_groups": len(test_groups),
    "overlapping_article_groups": 0,
}
(RESULTS / "split_report.json").write_text(
    json.dumps(split_report, indent=2), encoding="utf-8"
)

print("SPLITS CREATED")
print(json.dumps(split_report, indent=2))

# 3. Lead-3 extractive baseline on the test set
# Select the first three sentences as the predicted summary.
def lead3(article):
    article = re.sub(r"\s+", " ", str(article)).strip()
    sentences = re.split(r'(?<=[.!?])\s+', article)
    sentences = [s.strip() for s in sentences if s.strip()]
    return " ".join(sentences[:3])

scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], use_stemmer=True
)

scores = []
predictions = []

for row in test.itertuples(index=False):
    prediction = lead3(row.article_text)
    reference = str(row.human_summary).strip()

    if not prediction or not reference:
        continue

    result = scorer.score(reference, prediction)
    scores.append({
        metric: values.fmeasure
        for metric, values in result.items()
    })
    predictions.append({
        "source_row_id": row.source_row_id,
        "prediction": prediction,
        "reference": reference,
    })

if not scores:
    raise ValueError("No valid predictions to evaluate")

score_df = pd.DataFrame(scores)
metrics = {
    "baseline": "Lead-3 extractive summarization",
    "evaluated_examples": len(score_df),
    "rouge1_f1": float(score_df["rouge1"].mean()),
    "rouge2_f1": float(score_df["rouge2"].mean()),
    "rougeL_f1": float(score_df["rougeL"].mean()),
    "note": (
        "Provisional baseline. Test labels and source alignment "
        "have not been fully manually validated."
    ),
}

pd.DataFrame(predictions).to_csv(
    RESULTS / "baseline_lead3_predictions.csv", index=False
)
(RESULTS / "baseline_lead3_metrics.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)

print("\nLEAD-3 BASELINE RESULTS")
print(json.dumps(metrics, indent=2))
print("\nSaved split files:", OUT)
print("Saved metrics:", RESULTS / "baseline_lead3_metrics.json")
