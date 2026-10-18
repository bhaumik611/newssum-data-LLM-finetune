
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data/processed/splits/test.parquet"

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading {MODEL_NAME} on {DEVICE}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

df = pd.read_parquet(TEST_PATH).head(10)

for i, row in enumerate(df.itertuples(index=False), start=1):
    article = str(row.article_text).strip()

    inputs = tokenizer(
        article,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            num_beams=2,
            do_sample=False
        )

    summary = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    print(f"\n{'=' * 60}")
    print(f"ARTICLE {i}")
    print("REFERENCE:", str(row.human_summary)[:500])
    print("GENERATED:", summary)
