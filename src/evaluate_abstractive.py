
from pathlib import Path
import json
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data/processed/splits/test.parquet"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N = 500

df = pd.read_parquet(TEST_PATH).head(N)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], use_stemmer=True
)

def lead3(article):
    sentences = re.split(r"(?<=[.!?])\s+", str(article).strip())
    return " ".join(s for s in sentences[:3] if s)

def generate_summary(article):
    inputs = tokenizer(
        str(article),
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            num_beams=2,
            do_sample=False
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

predictions = []
model_scores = []
lead_scores = []

for i, row in enumerate(df.itertuples(index=False), start=1):
    reference = str(row.human_summary).strip()
    article = str(row.article_text).strip()

    if not article or not reference or reference.lower() == "nan":
        continue

    generated = generate_summary(article)
    baseline = lead3(article)

    if not generated.strip():
        continue

    model_scores.append(scorer.score(reference, generated))
    lead_scores.append(scorer.score(reference, baseline))

    predictions.append({
        "source_row_id": row.source_row_id,
        "reference": reference,
        "lead3_prediction": baseline,
        "distilbart_prediction": generated
    })

    if i % 50 == 0:
        print(f"Processed {i}/{len(df)} articles")

def average_scores(scores):
    return {
        metric: round(
            sum(s[metric].fmeasure for s in scores) / len(scores), 4
        )
        for metric in ["rouge1", "rouge2", "rougeL"]
    }

if not model_scores:
    raise RuntimeError("No valid predictions were generated.")

metrics = {
    "model": MODEL_NAME,
    "device": DEVICE,
    "test_subset_size": len(df),
    "evaluated_examples": len(model_scores),
    "distilbart": average_scores(model_scores),
    "lead3": average_scores(lead_scores),
    "note": "Preliminary 500-example comparison; not full-test results."
}

pd.DataFrame(predictions).to_csv(
    RESULTS / "abstractive_predictions_500.csv", index=False
)
(RESULTS / "abstractive_metrics_500.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)

print("\nFINAL COMPARISON")
print(json.dumps(metrics, indent=2))
