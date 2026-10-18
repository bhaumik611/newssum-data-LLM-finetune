
from pathlib import Path
import json
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "data" / "raw" / "NewsSumm Dataset.xlsx"
OUT_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

CLEAN_FILE = OUT_DIR / "newssumm_clean_candidate.parquet"
EXCLUDED_FILE = RESULTS_DIR / "excluded_missing_text.csv"
DUPLICATES_FILE = RESULTS_DIR / "duplicate_rows_removed.csv"
REPORT_FILE = RESULTS_DIR / "dataset_preparation_report.json"

# These dates are used only as a diagnostic window.
# Older dates are flagged, not automatically removed.
DIAGNOSTIC_START_YEAR = 2000
DIAGNOSTIC_END_YEAR = 2026

REQUIRED_COLUMNS = [
    "newspaper_name",
    "published_date",
    "headline",
    "article_text",
    "human_summary",
    "news_category",
]

TEXT_COLUMNS = [
    "newspaper_name",
    "headline",
    "article_text",
    "human_summary",
    "news_category",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_column_name(name):
    """Normalize column names, including embedded newlines."""
    return (
        str(name)
        .strip()
        .lower()
        .replace("\n", " ")
        .replace("\r", " ")
        .replace(" ", "_")
    )


def is_blank(series):
    """Identify missing values and strings containing only whitespace."""
    return series.isna() | series.astype("string").str.strip().eq("").fillna(True)


def json_safe(value):
    """Convert values into JSON-compatible Python values."""
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


# ============================================================
# MAIN PREPARATION PIPELINE
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{RAW_FILE}\n\n"
            "Check that NewsSumm Dataset.xlsx is inside data/raw/."
        )

    print("=" * 65)
    print("NEWS SUMM DATASET PREPARATION")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Load the Excel workbook
    # --------------------------------------------------------

    print("\n[1/8] Loading Excel dataset...")

    df = pd.read_excel(RAW_FILE)

    original_rows = len(df)

    # Excel row number, assuming row 1 contains the column headers.
    df.insert(0, "source_row_id", df.index + 2)

    # Normalize column names.
    df.columns = [normalize_column_name(c) for c in df.columns]

    # Guard against duplicate column names after normalization.
    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(
            f"Duplicate column names after normalization: {duplicates}"
        )

    missing_columns = [
        col for col in REQUIRED_COLUMNS if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    print(f"Original rows: {original_rows:,}")
    print(f"Original columns: {len(df.columns) - 1}")

    # --------------------------------------------------------
    # 2. Preserve original dates and normalize text fields
    # --------------------------------------------------------

    print("\n[2/8] Normalizing mixed data types...")

    # Preserve date values before converting them.
    # This is important for investigating suspicious source dates.
    df["published_date_original"] = (
        df["published_date"].astype("string")
    )

    # Normalize text columns to a consistent nullable string dtype.
    for col in TEXT_COLUMNS:
        df[col] = df[col].astype("string").str.strip()

        # Treat whitespace-only values as missing.
        df[col] = df[col].mask(df[col].eq(""))

    # Convert dates to datetime. Unparseable dates become NaT,
    # while the original value remains in published_date_original.
    df["published_date"] = pd.to_datetime(
        df["published_date"],
        errors="coerce",
        format="mixed",
    )

    # --------------------------------------------------------
    # 3. Identify rows with missing article text or summary
    # --------------------------------------------------------

    print("\n[3/8] Checking required article and summary text...")

    missing_article = is_blank(df["article_text"])
    missing_summary = is_blank(df["human_summary"])

    exclude_mask = missing_article | missing_summary

    excluded = df.loc[exclude_mask].copy()

    def exclusion_reason(row):
        reasons = []

        if pd.isna(row["article_text"]) or str(row["article_text"]).strip() == "":
            reasons.append("missing_or_blank_article_text")

        if pd.isna(row["human_summary"]) or str(row["human_summary"]).strip() == "":
            reasons.append("missing_or_blank_human_summary")

        return ";".join(reasons)

    if not excluded.empty:
        excluded["exclusion_reason"] = excluded.apply(
            exclusion_reason, axis=1
        )

    excluded.to_csv(EXCLUDED_FILE, index=False)

    # Keep rows that have both article text and reference summary.
    candidate = df.loc[~exclude_mask].copy()

    print(f"Rows excluded for missing article/summary: {len(excluded):,}")
    print(f"Rows retained for duplicate checking: {len(candidate):,}")

    # --------------------------------------------------------
    # 4. Flag date quality issues without deleting those rows
    # --------------------------------------------------------

    print("\n[4/8] Flagging date quality issues...")

    candidate["date_unparseable"] = candidate["published_date"].isna()

    candidate["date_outside_diagnostic_range"] = (
        candidate["published_date"].notna()
        & (
            (candidate["published_date"].dt.year < DIAGNOSTIC_START_YEAR)
            | (candidate["published_date"].dt.year > DIAGNOSTIC_END_YEAR)
        )
    )

    # These flags are diagnostic only. They do not remove rows.
    candidate["date_needs_review"] = (
        candidate["date_unparseable"]
        | candidate["date_outside_diagnostic_range"]
    )

    # --------------------------------------------------------
    # 5. Flag summary lengths that exceed article lengths
    # --------------------------------------------------------

    print("\n[5/8] Checking article and summary lengths...")

    candidate["article_chars"] = candidate["article_text"].str.len()
    candidate["summary_chars"] = candidate["human_summary"].str.len()

    candidate["summary_longer_than_article"] = (
        candidate["summary_chars"] > candidate["article_chars"]
    )

    # A longer summary is a review flag, not automatic proof of bad data.

    # --------------------------------------------------------
    # 6. Remove exact duplicate records
    # --------------------------------------------------------

    print("\n[6/8] Detecting exact duplicate records...")

    identity_columns = [
        "newspaper_name",
        "published_date_original",
        "headline",
        "article_text",
        "human_summary",
        "news_category",
    ]

    duplicate_mask = candidate.duplicated(
        subset=identity_columns,
        keep="first",
    )

    duplicates_removed = candidate.loc[duplicate_mask].copy()
    duplicates_removed.to_csv(DUPLICATES_FILE, index=False)

    before_deduplication = len(candidate)

    candidate = candidate.loc[~duplicate_mask].copy()
    candidate.reset_index(drop=True, inplace=True)

    exact_duplicates_removed = before_deduplication - len(candidate)

    print(f"Exact duplicate records removed: {exact_duplicates_removed:,}")
    print(f"Rows remaining: {len(candidate):,}")

    # --------------------------------------------------------
    # 7. Save processed dataset
    # --------------------------------------------------------

    print("\n[7/8] Saving cleaned candidate dataset...")

    candidate.to_parquet(
        CLEAN_FILE,
        index=False,
        engine="pyarrow",
    )

    # --------------------------------------------------------
    # 8. Save a reproducible preparation report
    # --------------------------------------------------------

    print("\n[8/8] Creating preparation report...")

    category_counts = (
        candidate["news_category"]
        .fillna("[MISSING]")
        .value_counts()
        .head(20)
        .to_dict()
    )

    report = {
        "input_file": str(RAW_FILE),
        "output_file": str(CLEAN_FILE),
        "original_rows": int(original_rows),
        "rows_excluded_missing_article_or_summary": int(len(excluded)),
        "rows_before_exact_deduplication": int(before_deduplication),
        "exact_duplicate_rows_removed": int(exact_duplicates_removed),
        "final_rows": int(len(candidate)),
        "final_columns": candidate.columns.tolist(),
        "missing_values_after_preparation": {
            col: int(count)
            for col, count in candidate.isna().sum().items()
        },
        "unparseable_dates": int(candidate["date_unparseable"].sum()),
        "dates_outside_diagnostic_range": int(
            candidate["date_outside_diagnostic_range"].sum()
        ),
        "rows_with_summary_longer_than_article": int(
            candidate["summary_longer_than_article"].sum()
        ),
        "top_20_categories": {
            str(k): int(v) for k, v in category_counts.items()
        },
        "diagnostic_date_window": {
            "start_year": DIAGNOSTIC_START_YEAR,
            "end_year": DIAGNOSTIC_END_YEAR,
            "note": "Flags only; dates outside this range are retained.",
        },
        "notes": [
            "Original date representations are preserved in published_date_original.",
            "Unparseable dates are stored as missing parsed dates and flagged.",
            "Rows with missing article text or human summary are excluded.",
            "Exact duplicates are removed using the six source fields.",
            "Summary length and date flags do not automatically exclude records.",
            "This is a cleaned candidate dataset, not a final research split.",
        ],
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, default=json_safe)

    print("\n" + "=" * 65)
    print("PREPARATION COMPLETE")
    print("=" * 65)
    print(f"Original rows:                    {original_rows:,}")
    print(f"Missing-text rows excluded:       {len(excluded):,}")
    print(f"Exact duplicate rows removed:     {exact_duplicates_removed:,}")
    print(f"Final rows:                       {len(candidate):,}")
    print(f"Unparseable dates:                {candidate['date_unparseable'].sum():,}")
    print(
        "Dates outside diagnostic range:  "
        f"{candidate['date_outside_diagnostic_range'].sum():,}"
    )
    print(
        "Summaries longer than articles:   "
        f"{candidate['summary_longer_than_article'].sum():,}"
    )
    print("\nSaved files:")
    print(f"  Clean candidate: {CLEAN_FILE}")
    print(f"  Excluded rows:   {EXCLUDED_FILE}")
    print(f"  Removed duplicates: {DUPLICATES_FILE}")
    print(f"  Report:          {REPORT_FILE}")


if __name__ == "__main__":
    main()
