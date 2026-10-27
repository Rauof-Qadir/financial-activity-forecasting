from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
REPORT_PATH = PROJECT_ROOT / "data" / "reports" / "data_audit.json"

REQUIRED_COLUMNS = {"step", "type", "amount"}
CHUNK_SIZE = 250_000


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Copy transactions.csv into data/raw/ first."
        )

    columns = set(pd.read_csv(DATA_PATH, nrows=0).columns)
    missing_columns = sorted(REQUIRED_COLUMNS - columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    total_rows = 0
    missing_values = {"step": 0, "type": 0, "amount": 0}
    invalid_step = 0
    invalid_amount = 0
    negative_amount = 0
    steps = set()
    type_counts = {}

    for chunk in pd.read_csv(
        DATA_PATH,
        usecols=["step", "type", "amount"],
        chunksize=CHUNK_SIZE,
    ):
        total_rows += len(chunk)

        for column in missing_values:
            missing_values[column] += int(
                chunk[column].isna().sum()
            )

        step = pd.to_numeric(chunk["step"], errors="coerce")
        amount = pd.to_numeric(chunk["amount"], errors="coerce")

        invalid_step += int(step.isna().sum())
        invalid_amount += int(amount.isna().sum())
        negative_amount += int((amount < 0).sum())

        steps.update(step.dropna().unique().tolist())

        counts = chunk["type"].value_counts(dropna=False)
        for transaction_type, count in counts.items():
            key = (
                str(transaction_type)
                if pd.notna(transaction_type)
                else "<MISSING>"
            )
            type_counts[key] = (
                type_counts.get(key, 0) + int(count)
            )

    numeric_steps = sorted(float(x) for x in steps)

    report = {
        "file": str(DATA_PATH),
        "total_rows": total_rows,
        "required_columns_found": sorted(REQUIRED_COLUMNS),
        "missing_values": missing_values,
        "invalid_step_values": invalid_step,
        "invalid_amount_values": invalid_amount,
        "negative_amount_rows": negative_amount,
        "unique_step_count": len(numeric_steps),
        "minimum_step": min(numeric_steps) if numeric_steps else None,
        "maximum_step": max(numeric_steps) if numeric_steps else None,
        "step_is_integer_valued": all(
            x.is_integer() for x in numeric_steps
        ),
        "transaction_type_counts": type_counts,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"\nAudit report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()