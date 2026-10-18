
from pathlib import Path
import json
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CLEAN_FILE = (
    ROOT / "data" / "processed" / "newssumm_clean_candidate.parquet"
)
REPORT_FILE = ROOT / "results" / "dataset_preparation_report.json"

REQUIRED_COLUMNS = [
    "source_row_id",
    "newspaper_name",
    "published_date",
    "published_date_original",
    "headline",
    "article_text",
    "human_summary",
    "news_category",
    "date_unparseable",
    "date_outside_diagnostic_range",
    "date_needs_review",
    "article_chars",
    "summary_chars",
    "summary_longer_than_article",
]

IDENTITY_COLUMNS = [
    "newspaper_name",
    "published_date_original",
    "headline",
    "article_text",
    "human_summary",
    "news_category",
]


def main():
    print("=" * 65)
    print("VERIFY PREPARED NEWS SUMM DATASET")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Check files exist
    # --------------------------------------------------------

    print("\n[1/7] Checking output files...")

    if not CLEAN_FILE.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found:\n{CLEAN_FILE}\n\n"
            "Run: python src/prepare_dataset.py"
        )

    if not REPORT_FILE.exists():
        raise FileNotFoundError(
            f"Preparation report not found:\n{REPORT_FILE}\n\n"
            "Run: python src/prepare_dataset.py"
        )

    print(f"Found dataset: {CLEAN_FILE}")
    print(f"Found report:  {REPORT_FILE}")

    # --------------------------------------------------------
    # 2. Load the saved Parquet file and report
    # --------------------------------------------------------

    print("\n[2/7] Loading prepared dataset...")

    df = pd.read_parquet(CLEAN_FILE)

    with open(REPORT_FILE, "r", encoding="utf-8") as file:
        report = json.load(file)

    print("Parquet file loaded successfully.")
    print(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # --------------------------------------------------------
    # 3. Verify expected columns
    # --------------------------------------------------------

    print("\n[3/7] Checking schema...")

    missing_columns = [
        col for col in REQUIRED_COLUMNS if col not in df.columns
    ]

    if missing_columns:
        print("FAIL: Missing expected columns:")
        for col in missing_columns:
            print(f"  - {col}")
    else:
        print("PASS: All expected columns exist.")

    print("\nColumn data types:")
    print(df.dtypes.to_string())

    # --------------------------------------------------------
    # 4. Check row counts and preparation arithmetic
    # --------------------------------------------------------

    print("\n[4/7] Checking row-count consistency...")

    original_rows = report["original_rows"]
    excluded_rows = report["rows_excluded_missing_article_or_summary"]
    duplicate_rows = report["exact_duplicate_rows_removed"]
    reported_final_rows = report["final_rows"]

    calculated_final_rows = (
        original_rows - excluded_rows - duplicate_rows
    )

    print(f"Original rows:             {original_rows:,}")
    print(f"Excluded rows:              {excluded_rows:,}")
    print(f"Exact duplicates removed:   {duplicate_rows:,}")
    print(f"Calculated final rows:      {calculated_final_rows:,}")
    print(f"Actual saved rows:           {len(df):,}")
    print(f"Rows stated in report:       {reported_final_rows:,}")

    row_count_ok = (
        calculated_final_rows == len(df) == reported_final_rows
    )

    if row_count_ok:
        print("PASS: Row counts are consistent.")
    else:
        print("FAIL: Row counts do not match.")

    # --------------------------------------------------------
    # 5. Check missing article text and summaries
    # --------------------------------------------------------

    print("\n[5/7] Checking required text fields...")

    for col in ["article_text", "human_summary"]:
        missing = df[col].isna()
        blank = df[col].astype("string").str.strip().eq("").fillna(False)
        problem_count = int((missing | blank).sum())

        print(f"{col}: {problem_count:,} missing/blank rows")

    article_ok = (
        df["article_text"].notna()
        & df["article_text"].astype("string").str.strip().ne("").fillna(False)
    )

    summary_ok = (
        df["human_summary"].notna()
        & df["human_summary"].astype("string").str.strip().ne("").fillna(False)
    )

    required_text_ok = bool((article_ok & summary_ok).all())

    print(
        "PASS: No missing/blank article or summary text."
        if required_text_ok
        else "FAIL: Some rows have missing/blank article or summary text."
    )

    # --------------------------------------------------------
    # 6. Check duplicates and date flags
    # --------------------------------------------------------

    print("\n[6/7] Checking duplicates and date flags...")

    remaining_exact_duplicates = int(
        df.duplicated(subset=IDENTITY_COLUMNS, keep=False).sum()
    )

    # This counts rows that belong to repeated groups.
    print(
        "Rows belonging to remaining exact-duplicate groups: "
        f"{remaining_exact_duplicates:,}"
    )

    no_exact_duplicates = remaining_exact_duplicates == 0

    print(
        "PASS: No exact duplicate groups remain."
        if no_exact_duplicates
        else "FAIL: Exact duplicate groups remain."
    )

    print("\nDate checks:")
    print(
        "Unparseable dates: "
        f"{int(df['date_unparseable'].sum()):,}"
    )
    print(
        "Dates outside diagnostic range: "
        f"{int(df['date_outside_diagnostic_range'].sum()):,}"
    )
    print(
        "Rows flagged for date review: "
        f"{int(df['date_needs_review'].sum()):,}"
    )

    print("\nSummary length checks:")
    print(
        "Summaries longer than articles: "
        f"{int(df['summary_longer_than_article'].sum()):,}"
    )

    # --------------------------------------------------------
    # 7. Print final verdict and sample rows
    # --------------------------------------------------------

    print("\n[7/7] Final verification...")

    checks = {
        "Expected schema": not missing_columns,
        "Row-count consistency": row_count_ok,
        "Required article/summary text": required_text_ok,
        "No remaining exact duplicates": no_exact_duplicates,
        "Parsed date column has datetime dtype": (
            pd.api.types.is_datetime64_any_dtype(df["published_date"])
        ),
        "Original date values preserved": (
            "published_date_original" in df.columns
        ),
    }

    for check_name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {check_name}")

    print("\nSample records:")
    sample_columns = [
        "source_row_id",
        "newspaper_name",
        "published_date_original",
        "published_date",
        "headline",
        "article_text",
        "human_summary",
        "news_category",
    ]

    existing_sample_columns = [
        col for col in sample_columns if col in df.columns
    ]

    with pd.option_context(
        "display.max_colwidth", 120,
        "display.width", 180,
    ):
        print(
            df[existing_sample_columns]
            .head(3)
            .to_string(index=False)
        )

    print("\nMissing values by column:")
    print(df.isna().sum().to_string())

    print("\nFinal result:")
    if all(checks.values()):
        print("PASS: All structural verification checks passed.")
        print(
            "Note: This verifies preparation consistency, not the "
            "factual correctness of every article-summary pair."
        )
    else:
        print(
            "WARNING: One or more checks failed. Review the output "
            "before using the dataset for experiments."
        )


if __name__ == "__main__":
    main()
