
from pathlib import Path
import json
import pandas as pd

DATA_DIR = Path("data/raw")
SUPPORTED = {".xlsx", ".xlsm", ".csv", ".json", ".jsonl", ".parquet"}

files = [
    p for p in DATA_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in SUPPORTED
]

if not files:
    print("No supported files found in:", DATA_DIR.resolve())
    raise SystemExit(1)

for path in files:
    print(f"\n{'=' * 70}")
    print("FILE:", path.name)

    try:
        if path.suffix.lower() in {".xlsx", ".xlsm"}:
            # Inspect all workbook sheets
            sheets = pd.read_excel(path, sheet_name=None)

            for sheet_name, df in sheets.items():
                print(f"\nSHEET: {sheet_name}")
                print("Shape:", df.shape)
                print("Columns:", df.columns.tolist())
                print("\nFirst 3 rows:")
                print(df.head(3).to_string(index=False))

        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path, nrows=5)
            print("Columns:", df.columns.tolist())
            print(df.to_string(index=False))

        elif path.suffix.lower() == ".jsonl":
            df = pd.read_json(path, lines=True, nrows=5)
            print("Columns:", df.columns.tolist())
            print(df.to_string(index=False))

        elif path.suffix.lower() == ".json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            print(json.dumps(data[:3] if isinstance(data, list)
                             else data, indent=2, ensure_ascii=False)[:5000])

        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path).head(3)
            print("Columns:", df.columns.tolist())
            print(df.to_string(index=False))

    except Exception as exc:
        print("Inspection failed:", repr(exc))
