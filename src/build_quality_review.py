
from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "newssumm_clean_candidate.parquet"
OUT = ROOT / "results"

OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(INPUT)

# ------------------------------------------------------------
# 1. Detect articles that appear to contain only a link or
#    very little usable article text.
# ------------------------------------------------------------

article = df["article_text"].fillna("").astype(str).str.strip()
summary = df["human_summary"].fillna("").astype(str).str.strip()
headline = df["headline"].fillna("").astype(str).str.strip()

df["article_chars_review"] = article.str.len()

df["flag_url_only_article"] = article.str.fullmatch(
    r"https?://\S+/?",
    case=False,
).fillna(False)

df["flag_very_short_article"] = article.str.len() < 200

# ------------------------------------------------------------
# 2. Detect summaries containing task instructions,
#    refusals, or obvious extraction contamination.
# ------------------------------------------------------------

instruction_pattern = re.compile(
    r"\b(?:"
    r"minimum\s+\d+\s+words?"
    r"|maximum\s+\d+\s+words?"
    r"|write\s+a\s+summary"
    r"|human summary for article"
    r"|content in detail of news article"
    r"|article in detail from data source"
    r"|explain the news in your own words"
    r"|provide a summary"
    r")\b",
    flags=re.IGNORECASE,
)

refusal_pattern = re.compile(
    r"\b(?:"
    r"i can't summarize"
    r"|i cannot summarize"
    r"|i don't have enough context"
    r"|i do not have enough context"
    r"|i'm unable to provide a summary"
    r"|the search results do not contain"
    r"|i cannot provide verbatim excerpts"
    r")\b",
    flags=re.IGNORECASE,
)

df["flag_summary_instruction"] = summary.str.contains(
    instruction_pattern,
    na=False,
)

df["flag_summary_refusal_or_error"] = summary.str.contains(
    refusal_pattern,
    na=False,
)

# Detect summaries that are predominantly a copied URL.
df["flag_summary_url_only"] = summary.str.fullmatch(
    r"https?://\S+/?",
    case=False,
).fillna(False)

# ------------------------------------------------------------
# 3. Flag suspicious headlines and length relationships.
#    These are review signals, not automatic exclusions.
# ------------------------------------------------------------

df["flag_missing_headline"] = headline.eq("")

df["flag_generic_headline"] = headline.str.casefold().isin(
    {
        "mint",
        "the hindu",
        "business standard",
        "indian express",
        "economic times",
        "times of india",
    }
)

df["summary_chars_review"] = summary.str.len()

df["flag_summary_longer_than_article"] = (
    df["summary_chars_review"] > df["article_chars_review"]
)

# Very long summaries can be suspicious, but length alone does
# not prove a summary is wrong.
df["flag_extreme_summary_length"] = (
    df["summary_chars_review"] > 3000
)

# ------------------------------------------------------------
# 4. Combine flags without silently deleting records.
# ------------------------------------------------------------

flag_columns = [
    "flag_url_only_article",
    "flag_very_short_article",
    "flag_summary_instruction",
    "flag_summary_refusal_or_error",
    "flag_summary_url_only",
    "flag_missing_headline",
    "flag_generic_headline",
    "flag_summary_longer_than_article",
    "flag_extreme_summary_length",
]

df["quality_flag_count"] = df[flag_columns].sum(axis=1)

df["review_priority"] = "no_rule_triggered"

# High-priority candidates have a particularly strong signal
# of unusable or contaminated source/target text.
high_priority = (
    df["flag_url_only_article"]
    | df["flag_summary_url_only"]
    | df["flag_summary_instruction"]
    | df["flag_summary_refusal_or_error"]
    | df["flag_generic_headline"]
)

df.loc[high_priority, "review_priority"] = "high"

medium_priority = (
    ~high_priority
    & (
        df["flag_very_short_article"]
        | df["flag_missing_headline"]
        | df["flag_summary_longer_than_article"]
        | df["flag_extreme_summary_length"]
    )
)

df.loc[medium_priority, "review_priority"] = "medium"

# Do not automatically drop rows. Human review is the next step.
review_columns = [
    "source_row_id",
    "newspaper_name",
    "published_date_original",
    "headline",
    "article_text",
    "human_summary",
    "news_category",
    "article_chars_review",
    "summary_chars_review",
    "quality_flag_count",
    "review_priority",
] + flag_columns

review = df[review_columns].copy()

review = review.sort_values(
    ["review_priority", "quality_flag_count"],
    ascending=[True, False],
)

review.to_csv(
    OUT / "quality_review_all_records.csv",
    index=False,
)

high = review[review["review_priority"] == "high"]
medium = review[review["review_priority"] == "medium"]

high.to_csv(
    OUT / "quality_review_high_priority.csv",
    index=False,
)

medium.to_csv(
    OUT / "quality_review_medium_priority.csv",
    index=False,
)

print("=" * 65)
print("QUALITY REVIEW TABLES CREATED")
print("=" * 65)
print(f"Total records: {len(df):,}")
print(f"High-priority review: {len(high):,}")
print(f"Medium-priority review: {len(medium):,}")
print(
    "No rule triggered: "
    f"{(review['review_priority'] == 'no_rule_triggered').sum():,}"
)

print("\nIndividual flag counts:")
for col in flag_columns:
    print(f"{col}: {int(df[col].sum()):,}")

print("\nSaved:")
print(OUT / "quality_review_all_records.csv")
print(OUT / "quality_review_high_priority.csv")
print(OUT / "quality_review_medium_priority.csv")
