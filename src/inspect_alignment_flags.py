
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

low_path = RESULTS / "low_headline_article_similarity_sample.csv"
instruction_path = RESULTS / "summary_instruction_flags.csv"

# Inspect the 20 lowest-similarity headline/article pairs.
low = pd.read_csv(low_path)

columns = [
    "source_row_id",
    "newspaper_name",
    "published_date_original",
    "headline",
    "headline_article_tfidf_similarity",
    "article_text",
    "human_summary",
]

columns = [c for c in columns if c in low.columns]

print("=" * 70)
print("20 LOWEST-SIMILARITY RECORDS")
print("=" * 70)

for _, row in low.head(20).iterrows():
    print(f"\nSource row: {row.get('source_row_id')}")
    print(f"Publisher: {row.get('newspaper_name')}")
    print(f"Date: {row.get('published_date_original')}")
    print(f"Similarity: {row.get('headline_article_tfidf_similarity')}")
    print(f"Headline: {row.get('headline')}")
    print(f"Article: {str(row.get('article_text'))[:700]}")
    print(f"Summary: {str(row.get('human_summary'))[:400]}")
    print("-" * 70)

# Inspect possible annotation instructions.
if instruction_path.exists():
    flagged = pd.read_csv(instruction_path)

    print("\n" + "=" * 70)
    print("POSSIBLE SUMMARY INSTRUCTIONS")
    print("=" * 70)

    for _, row in flagged.head(30).iterrows():
        print(f"\nSource row: {row.get('source_row_id')}")
        print(f"Headline: {row.get('headline')}")
        print(f"Summary: {row.get('human_summary')}")
        print("-" * 70)
