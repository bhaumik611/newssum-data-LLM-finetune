
from pathlib import Path
import re
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "newssumm_clean_candidate.parquet"
OUT = ROOT / "results"

OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(INPUT)

# ------------------------------------------------------------
# 1. Detect annotation instructions accidentally included
#    in the human summary.
# ------------------------------------------------------------


instruction_pattern = re.compile(
    r"\b(?:"
    r"minimum\s+\d+\s+words?"
    r"|maximum\s+\d+\s+words?"
    r"|write\s+a\s+summary"
    r"|summarize\s+the\s+article"
    r"|provide\s+a\s+summary"
    r")\b",
    flags=re.IGNORECASE,
)

df["summary_has_instruction"] = (
    df["human_summary"]
    .str.contains(instruction_pattern, na=False)
)

instruction_rows = df.loc[
    df["summary_has_instruction"],
    [
        "source_row_id",
        "newspaper_name",
        "headline",
        "article_text",
        "human_summary",
        "news_category",
    ],
].copy()

instruction_rows.to_csv(
    OUT / "summary_instruction_flags.csv",
    index=False,
)

# ------------------------------------------------------------
# 2. Build a deterministic sample for manual inspection.
#    We use fixed random_state for reproducibility.
# ------------------------------------------------------------

sample_size = min(2000, len(df))

sample = df.sample(
    n=sample_size,
    random_state=42,
).copy()

# ------------------------------------------------------------
# 3. Compare headline and article using TF-IDF cosine similarity.
#
# This is a screening heuristic, not a definitive relevance test.
# Short headlines and articles can legitimately have low scores.
# ------------------------------------------------------------

headline_text = sample["headline"].fillna("").astype(str)
article_text = sample["article_text"].fillna("").astype(str)

combined_text = pd.concat(
    [headline_text, article_text],
    ignore_index=True,
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=100_000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)

try:
    matrix = vectorizer.fit_transform(combined_text)

    n = len(sample)
    headline_vectors = matrix[:n]
    article_vectors = matrix[n:]

    # Row-wise cosine similarity for each headline/article pair.
    similarities = (
        headline_vectors.multiply(article_vectors)
        .sum(axis=1)
        .A1
    )

    sample["headline_article_tfidf_similarity"] = similarities

except ValueError as exc:
    raise RuntimeError(
        "TF-IDF could not build a vocabulary. "
        "Inspect whether the sample contains meaningful text."
    ) from exc

# A low score is a review flag only, not an automatic exclusion.
low_similarity = sample.loc[
    sample["headline_article_tfidf_similarity"] < 0.05,
    [
        "source_row_id",
        "newspaper_name",
        "published_date_original",
        "headline",
        "article_text",
        "human_summary",
        "headline_article_tfidf_similarity",
    ],
].sort_values("headline_article_tfidf_similarity")

low_similarity.to_csv(
    OUT / "low_headline_article_similarity_sample.csv",
    index=False,
)

# Save all scored sample records for threshold inspection.
sample[
    [
        "source_row_id",
        "newspaper_name",
        "published_date_original",
        "headline",
        "human_summary",
        "headline_article_tfidf_similarity",
    ]
].to_csv(
    OUT / "headline_article_similarity_sample.csv",
    index=False,
)

# ------------------------------------------------------------
# 4. Produce a concise report.
# ------------------------------------------------------------

report = {
    "dataset_rows": int(len(df)),
    "sample_size": int(sample_size),
    "summary_instruction_rows_entire_dataset": int(
        df["summary_has_instruction"].sum()
    ),
    "sample_low_similarity_threshold": 0.05,
    "sample_low_similarity_rows": int(
        (sample["headline_article_tfidf_similarity"] < 0.05).sum()
    ),
    "sample_similarity_quantiles": {
        str(q): float(sample["headline_article_tfidf_similarity"].quantile(q))
        for q in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    },
    "warning": (
        "TF-IDF similarity is a screening heuristic. "
        "Low similarity does not prove a record is invalid."
    ),
}

with open(
    OUT / "alignment_audit_report.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("=" * 60)
print("ALIGNMENT AUDIT COMPLETE")
print("=" * 60)
print(f"Dataset rows: {len(df):,}")
print(
    "Rows with possible summary instructions: "
    f"{df['summary_has_instruction'].sum():,}"
)
print(f"Sample size: {sample_size:,}")
print(
    "Low-similarity sample rows: "
    f"{(sample['headline_article_tfidf_similarity'] < 0.05).sum():,}"
)
print("\nSimilarity quantiles:")
print(
    sample["headline_article_tfidf_similarity"]
    .quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    .to_string()
)
print("\nReports saved in:", OUT)
