
from pathlib import Path
import json
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
OUT_DIR = ROOT / "models/distilbart-news"
TRAIN_PATH = ROOT / "data/processed/splits/train.parquet"
VAL_PATH = ROOT / "data/processed/splits/validation.parquet"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 42

print("Device:", DEVICE)
print("Loading dataset...")

train_df = pd.read_parquet(TRAIN_PATH).sample(
    n=2000, random_state=SEED
).reset_index(drop=True)

val_df = pd.read_parquet(VAL_PATH).sample(
    n=200, random_state=SEED
).reset_index(drop=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def tokenize_batch(batch):
    inputs = tokenizer(
        batch["article_text"],
        max_length=512,
        truncation=True,
    )
    targets = tokenizer(
        text_target=batch["human_summary"],
        max_length=128,
        truncation=True,
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

from datasets import Dataset

train_ds = Dataset.from_pandas(
    train_df[["article_text", "human_summary"]]
)
val_ds = Dataset.from_pandas(
    val_df[["article_text", "human_summary"]]
)

train_ds = train_ds.map(
    tokenize_batch, batched=True, remove_columns=train_ds.column_names
)
val_ds = val_ds.map(
    tokenize_batch, batched=True, remove_columns=val_ds.column_names
)

args = Seq2SeqTrainingArguments(
    output_dir=str(OUT_DIR / "checkpoints"),
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=25,
    predict_with_generate=False,
    fp16=False,
    bf16=False,
    report_to="none",
    use_cpu=(DEVICE == "cpu"),
    dataloader_num_workers=0,
    save_total_limit=1,
    seed=SEED,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model
    ),
    processing_class=tokenizer,
)

print("Starting fine-tuning...")
trainer.train()

OUT_DIR.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(OUT_DIR))
tokenizer.save_pretrained(str(OUT_DIR))

result = {
    "model": MODEL_NAME,
    "device": DEVICE,
    "train_examples": len(train_ds),
    "validation_examples": len(val_ds),
    "epochs": 1,
    "training_metrics": trainer.state.log_history,
}

(OUT_DIR / "training_report.json").write_text(
    json.dumps(result, indent=2, default=str),
    encoding="utf-8",
)

print("Training complete.")
print("Saved model to:", OUT_DIR)
