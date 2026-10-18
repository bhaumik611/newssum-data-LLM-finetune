# NewsSumm-Research

An experimental news-summarization pipeline for examining a NewsSumm
spreadsheet, preparing article/summary pairs, reviewing data-quality risks,
creating leakage-aware splits, and comparing extractive and abstractive
summarization approaches.

The project is research code, not a production training package. In
particular, the quality checks are screening tools, and the generated metrics
should be treated as provisional until the source data and reference summaries
have been reviewed.

## What the project does

Each usable record is intended to pair a news article with a human-written
summary. The article is the model input; the human summary is the target during
fine-tuning and the reference during evaluation. The scripts first protect and
inspect these pairs, then build datasets and evaluate summaries against the
reference.

### Dataset fields and their purpose

The preparation code expects these source columns:

| Field | Why it is useful |
| --- | --- |
| `article_text` | The source document the summarizer reads. This is the model input and is essential for training and generating summaries. |
| `human_summary` | The reference/target summary. The model learns to produce text like this, and evaluation compares generated summaries with it. |
| `headline` | Useful for human inspection and a lightweight headline/article relevance screen. It is not the fine-tuning target in the current scripts. |
| `newspaper_name` | Source provenance for auditing and examining publisher-level data patterns. It is retained with prepared records, but is not used as model input by the current trainer. |
| `published_date` | Supports date-range and parsing audits. The original value is retained separately so date conversion does not erase source evidence. |
| `news_category` | Retained for category-level inspection and possible future stratified analysis; the current training/evaluation scripts do not use it as an input feature. |
| `source_row_id` | Added during preparation to trace a processed record back to its spreadsheet row and consistently apply review decisions. |

The article-summary pairing is the core learning signal. The metadata fields
help investigate and reproduce the dataset, but are not currently fed into
DistilBART.

## End-to-end flow

```text
NewsSumm Dataset.xlsx
        |
        +--> Inspect schema, missingness, dates, duplicates, and text
        |
        +--> Prepare candidate data
        |      - preserve spreadsheet row identity and original dates
        |      - normalize text and parse dates
        |      - exclude rows missing article text or human summary
        |      - remove exact duplicate records
        |      - flag (not automatically drop) date/length anomalies
        |
        +--> Verify prepared structure
        |
        +--> Audit quality and alignment; build review tables
        |      - inspect likely contaminated or unusable examples
        |      - review proposed cleaning decisions
        |
        +--> Create initial training dataset
        |
        +--> Group-split on normalized article text
        |      +--> Train / validation / test
        |      +--> Lead-3 baseline on test
        |
        +--> Fine-tune DistilBART on train; validate during training
        |
        +--> Evaluate baseline/model summaries with ROUGE
```

### Recommended sequence

Run commands from the repository root. Some diagnostic scripts use paths
relative to the current directory, so running them from another directory may
not find the data.

1. **Place the source workbook locally.** Put the permitted NewsSumm workbook
   at `data/raw/NewsSumm Dataset.xlsx`. The raw data is intentionally excluded
   by `.gitignore`; do not commit it unless you have permission and have
   checked its license and privacy implications.
2. **Inspect the environment and raw data.**

   ```bash
   python src/check_environment.py
   python src/inspect_dataset.py
   python src/audit_dataset.py
   python src/debug_parquet.py
   python src/investigate_quality.py
   ```

   These scripts are exploratory diagnostics; they do not create the model
   training split. `audit_dataset.py` writes `results/dataset_audit.json`, and
   `investigate_quality.py` writes date and duplicate-conflict diagnostics.
3. **Prepare and verify the candidate table.**

   ```bash
   python src/prepare_dataset.py
   python src/verify_dataset.py
   ```

   Preparation excludes rows that lack article text or a human summary and
   removes exact duplicates based on the six source fields. Date and
   summary-length anomalies are flags, not automatic exclusion rules.
4. **Generate data-quality review material.**

   ```bash
   python src/build_quality_review.py
   python src/audit_alignment.py
   python src/inspect_alignment_flags.py
   python src/create_cleaning_decisions.py
   python src/inspect_exclusion_candidates.py
   ```

   Review the generated rows against the source workbook before accepting
   exclusions. These rules identify candidates; a rule trigger is not proof
   that a record is invalid.
5. **Create the provisional modeling dataset.** Review and update
   `results/cleaning_decisions.csv` before running:

   ```bash
   python src/create_initial_training_dataset.py
   ```

   This builder excludes rows whose `proposed_action` is
   `exclude_candidate`; it does not use the `human_decision` or `human_notes`
   fields to override that proposed action. If human review changes a
   candidate's disposition, update `proposed_action` accordingly before
   building the dataset. The output is explicitly provisional, not a claim
   that all remaining pairs are semantically correct.
6. **Audit duplicates and build splits/baseline.**

   ```bash
   python src/audit_duplicate_leakage.py
   python src/inspect_conflicting_duplicates.py
   python src/finalize_and_evaluate.py
   ```

   The splitter hashes normalized article text and groups identical articles
   together, reducing the risk that the same article appears in more than one
   partition. Its 80/10/10 proportions are approximate because groups cannot
   be split. The final script writes the three Parquet splits and evaluates a
   Lead-3 extractive baseline on the test partition.
7. **Train and evaluate.**

   ```bash
   python src/train_distilbart.py
   python src/test_abstractive.py
   python src/evaluate_abstractive.py
   python src/evaluate_textrank.py
   ```

   Training fine-tunes `sshleifer/distilbart-cnn-12-6`; the current script
   samples 2,000 training rows and 200 validation rows, runs one epoch, and
   saves the model and tokenizer below `models/distilbart-news/`.

   **Important evaluation detail:** `evaluate_abstractive.py` and
   `test_abstractive.py` currently load the named Hugging Face checkpoint
   directly. They do not load the fine-tuned artifact under
   `models/distilbart-news/`. `evaluate_abstractive.py` compares the
   checkpoint's output with Lead-3 on the first 500 test rows; it is not an
   evaluation of the locally fine-tuned model. `evaluate_textrank.py` evaluates
   TextRank on the test partition.

## Source code guide

All scripts below are in `src/`. Together, the diagnostics and review tools
exist to understand what the dataset contains and reduce avoidable training
and evaluation errors. They are standalone scripts rather than a single
command-line application.

### Source inspection and preparation

| File | What it does | Why it is needed for this dataset |
| --- | --- | --- |
| `check_environment.py` | Prints Python, platform, NumPy, pandas, scikit-learn, PyTorch, Transformers, and Apple MPS availability; tries a small MPS matrix operation when available. | Confirms which runtime/device is being used before expensive model downloads or training. |
| `inspect_dataset.py` | Recursively previews supported spreadsheet, CSV, JSON, JSONL, and Parquet files under `data/raw/`. | Provides an initial, format-aware look at columns and sample values without changing the data. |
| `audit_dataset.py` | Checks the NewsSumm workbook schema, missing/blank values, dates, duplicates, text lengths, category/publisher counts, and writes a JSON report. | Establishes the source dataset's shape and common data-quality issues before transformation. |
| `debug_parquet.py` | Reports columns that contain mixed Python value types in the raw workbook. | Helps diagnose inconsistent spreadsheet cell types that can cause conversion or downstream processing problems. |
| `investigate_quality.py` | Examines date anomalies, normalized duplicate headlines/articles, repeated article texts with different summaries, category normalization, and basic summary/article length issues. | Surfaces suspicious patterns and conflicts that basic schema checks do not explain. It is diagnostic; it does not clean or exclude rows. |
| `prepare_dataset.py` | Normalizes column names and text, adds spreadsheet `source_row_id`, preserves original date strings, parses dates, excludes missing article/summary pairs, flags selected anomalies, removes exact duplicate source rows, and writes a Parquet candidate plus reports. | Creates a reproducible, structurally consistent starting point while keeping an audit trail and avoiding automatic removal based on weak heuristics. |
| `verify_dataset.py` | Checks the prepared Parquet schema and report arithmetic, required text, duplicates, date flags, and basic invariants. | Detects preparation/output inconsistencies before more expensive human review or model work. It does not establish semantic correctness. |

### Quality review and duplicate investigation

| File | What it does | Why it is needed for this dataset |
| --- | --- | --- |
| `build_quality_review.py` | Flags URL-only/short articles, suspicious summaries, instruction/refusal text, missing or generic headlines, and unusual length relationships; writes full and prioritized review CSVs. | Creates manageable, inspectable review candidates for potentially unusable article-summary pairs without silently deleting data. |
| `audit_alignment.py` | Flags possible instructions in summaries, samples up to 2,000 rows, calculates TF-IDF headline/article similarity, and writes a JSON report and CSVs. | Screens for possible article/headline mismatch and annotation contamination. Similarity is only a heuristic and cannot prove summary/article alignment. |
| `inspect_alignment_flags.py` | Prints examples among the low headline/article similarity results and possible instruction-containing summaries. | Makes the alignment audit's warning cases easier for a person to inspect. |
| `create_cleaning_decisions.py` | Converts quality flags into proposed retain, manual-review, or exclusion-candidate actions and creates blank human-review fields. | Gives a traceable decision table. The proposed actions must be checked; this script does not itself establish that an example is bad. |
| `inspect_exclusion_candidates.py` | Samples proposed exclusion candidates by reason and writes a smaller CSV for inspection. | Helps prioritize manual checks across different exclusion reasons rather than only examining the first rows. |
| `create_initial_training_dataset.py` | Applies `exclude_candidate` actions to the prepared candidate, checks row identities and text, and writes the initial Parquet, exclusions, and report. | Converts the reviewed decision table into a modeling input while preserving excluded records and the reasons for them. |
| `audit_duplicate_leakage.py` | Hashes normalized article text in the initial dataset, groups repeated articles, counts different summaries/headlines, and writes a detailed duplicate report. | Detects repeated source text that can distort evaluation or leak across data splits. It does not remove records. |
| `inspect_conflicting_duplicates.py` | Samples repeated-article groups with more than one normalized summary and prints all records in each sampled group. | Helps determine whether repeated source texts represent legitimate variant references or conflicting labels. |

### Splitting, modeling, and evaluation

| File | What it does | Why it is needed for this dataset |
| --- | --- | --- |
| `finalize_and_evaluate.py` | Rechecks required fields, groups rows by normalized article-text hash, makes approximate 80/10/10 train/validation/test splits, asserts no exact article group crosses splits, and evaluates a Lead-3 baseline with ROUGE. | Produces repeatable partitions while preventing exact duplicate articles from inflating held-out scores, then establishes an extractive comparison point. |
| `train_distilbart.py` | Tokenizes `article_text`/`human_summary` for DistilBART, fine-tunes a sampled subset, and saves checkpoints, tokenizer, model, and a training report. | Trains an abstractive model to map news article text to human summary text. |
| `test_abstractive.py` | Generates and prints summaries for up to 10 test rows using the pretrained DistilBART checkpoint. | A quick qualitative smoke check for generation and article/reference alignment; it does not score the full test set or load the local fine-tuned model. |
| `evaluate_abstractive.py` | Generates summaries for the first 500 test rows using the pretrained DistilBART checkpoint; compares ROUGE with Lead-3 and saves predictions/metrics. | Provides a preliminary, same-subset comparison between a pretrained abstractive model and a simple extractive baseline. |
| `evaluate_textrank.py` | Runs TextRank for three sentences on the test partition, scores with ROUGE, and saves predictions/metrics. | Adds a non-neural extractive baseline to contextualize the summarization task. |

## Inputs, outputs, and directory layout

```text
.
├── data/
│   ├── raw/                 # Local source workbook (not committed)
│   └── processed/           # Generated Parquet candidate and splits
├── models/                  # Fine-tuned weights, tokenizer, checkpoints
├── notebooks/               # Reserved for exploratory notebooks
├── results/                 # Audit reports, review CSVs, metrics, predictions
└── src/                     # Dataset, audit, training, and evaluation scripts
```

The directories under `data/processed/`, `models/`, and `results/` contain
generated or data-derived files and are ignored by Git. Keep the raw workbook
in its expected local path. To share reproducible results, prefer sharing
small, reviewed reports or metric summaries rather than prediction CSVs that
may contain article or summary text.

Key generated files include:

| Output | Created by | Meaning |
| --- | --- | --- |
| `data/processed/newssumm_clean_candidate.parquet` | `prepare_dataset.py` | Structurally prepared records before quality-review decisions. |
| `data/processed/newssumm_initial_training.parquet` | `create_initial_training_dataset.py` | Provisional records after applying exclusion candidates. |
| `data/processed/splits/train.parquet`, `validation.parquet`, `test.parquet` | `finalize_and_evaluate.py` | Group-aware model/evaluation partitions. |
| `results/dataset_preparation_report.json` | `prepare_dataset.py` | Counts, schema, flags, and preparation notes. |
| `results/cleaning_decisions.csv` | `create_cleaning_decisions.py` | Per-record proposed disposition and human review fields. |
| `results/split_report.json` | `finalize_and_evaluate.py` | Seed, split sizes, group counts, and overlap check. |
| `results/baseline_lead3_metrics.json` | `finalize_and_evaluate.py` | Lead-3 ROUGE scores on the test partition. |
| `results/abstractive_metrics_500.json` | `evaluate_abstractive.py` | Preliminary 500-row checkpoint-vs-Lead-3 ROUGE comparison. |
| `results/baseline_textrank_metrics.json` | `evaluate_textrank.py` | TextRank ROUGE results on the test partition. |
| `models/distilbart-news/` | `train_distilbart.py` | Local fine-tuned model, tokenizer, training report, and checkpoints. |

## Setup

There is currently no dependency lockfile or `requirements.txt`. Install the
packages needed by the scripts in a virtual environment; pin and record
versions before treating an experiment as reproducible.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pandas openpyxl pyarrow scikit-learn rouge-score sumy torch transformers datasets
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

The code selects Apple MPS when PyTorch reports it available and otherwise
uses CPU. The training script samples only a small subset for a quick
experiment; actual memory requirements depend on the device and loaded model.
The first use of Hugging Face model APIs may download checkpoint files into
the user's external Hugging Face cache.

## Evaluation and research caveats

- ROUGE measures lexical overlap; it does not by itself establish factual
  accuracy, informativeness, or faithfulness to the article.
- The headline/article TF-IDF score is not a semantic alignment test, and
  date/length flags are not proof of bad records.
- Exact text hashes catch normalized-identical articles, not near-duplicate
  or syndicated articles with small edits.
- The split is grouped by exact normalized article text, but other correlated
  examples may still occur across partitions.
- `train_distilbart.py` uses 2,000 train and 200 validation examples, one
  epoch, 512-token input truncation, and 128-token target truncation. It is a
  bounded experiment, not full-dataset training.
- The abstractive evaluation currently uses the pretrained checkpoint rather
  than the fine-tuned artifact; distinguish those results when reporting
  experiments.
- Check the NewsSumm dataset's license, attribution, and permitted use before
  redistribution or publication.
# newssum-data-LLM-finetune
