
from pathlib import Path
import json
import re

import pandas as pd
from rouge_score import rouge_scorer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data/processed/splits/test.parquet"
RESULTS = ROOT / "results"

RESULTS.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(TEST)
scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], use_stemmer=True
)

def textrank_summary(article, sentence_count=3):
    article = re.sub(r"\s+", " ", str(article)).strip()

    if not article:
        return ""

    try:
        parser = PlaintextParser.from_string(article, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        sentences = summarizer(parser.document, sentence_count)
        return " ".join(str(sentence) for sentence in sentences)
    except Exception:
        # Return an empty prediction rather than silently using Lead-3.
        return ""

scores = []
predictions = []

# Run on the same test rows as Lead-3.
for row in df.itertuples(index=False):
    prediction = textrank_summary(row.article_text, sentence_count=3)
    reference = str(row.human_summary).strip()

    if not prediction or not reference:
        continue

    result = scorer.score(reference, prediction)
    scores.append({
        metric: value.fmeasure
        for metric, value in result.items()
    })

    predictions.append({
        "source_row_id": row.source_row_id,
        "prediction": prediction,
        "reference": reference,
    })

if not scores:
    raise RuntimeError("TextRank produced no evaluable predictions.")

score_df = pd.DataFrame(scores)

metrics = {
    "baseline": "TextRank extractive summarization",
    "evaluated_examples": len(score_df),
    "rouge1_f1": float(score_df["rouge1"].mean()),
    "rouge2_f1": float(score_df["rouge2"].mean()),
    "rougeL_f1": float(score_df["rougeL"].mean()),
    "failed_or_empty_predictions": len(df) - len(score_df),
    "note": "Provisional scores; reference summaries are not fully validated.",
}

pd.DataFrame(predictions).to_csv(
    RESULTS / "baseline_textrank_predictions.csv", index=False
)
(RESULTS / "baseline_textrank_metrics.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)

print("TEXTRANK RESULTS")
print(json.dumps(metrics, indent=2))

# Display comparison with the saved Lead-3 metrics.
lead3_path = RESULTS / "baseline_lead3_metrics.json"
if lead3_path.exists():
    lead3 = json.loads(lead3_path.read_text(encoding="utf-8"))

    print("\nBASELINE COMPARISON")
    print(f"{'Metric':<12} {'Lead-3':>12} {'TextRank':>12}")
    for key in ["rouge1_f1", "rouge2_f1", "rougeL_f1"]:
        print(
            f"{key:<12} "
            f"{lead3[key]:>12.4f} "
            f"{metrics[key]:>12.4f}"
        )
