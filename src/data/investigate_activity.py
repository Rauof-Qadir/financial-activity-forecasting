from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "reports" / "hourly_activity_audit.csv"
)


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")

    hourly_parts = []

    for chunk in pd.read_csv(
        RAW_PATH,
        usecols=["step", "amount"],
        chunksize=250_000,
    ):
        chunk["step"] = pd.to_numeric(
            chunk["step"], errors="raise"
        )
        chunk["amount"] = pd.to_numeric(
            chunk["amount"], errors="raise"
        )

        hourly = chunk.groupby("step").agg(
            transaction_count=("amount", "size"),
            total_amount=("amount", "sum"),
        )

        hourly_parts.append(hourly)

    hourly = pd.concat(hourly_parts).groupby(level=0).sum()
    hourly.index = hourly.index.astype(int)
    hourly.index.name = "step"
    hourly = hourly.sort_index()

    hourly["day_index"] = (hourly.index - 1) // 24 + 1

    # Verify each day contains the expected 24 hourly steps.
    step_counts = hourly.groupby("day_index").size()

    print("Hourly observations:", len(hourly))
    print("Days represented:", hourly["day_index"].nunique())
    print("Days with fewer than 24 observed steps:")
    print(step_counts[step_counts != 24].to_string())

    print("\nDaily transaction counts rebuilt from raw data:")
    daily_counts = hourly.groupby("day_index")[
        "transaction_count"
    ].sum()
    print(daily_counts.head(10).to_string())

    # Inspect each hour contributing to days 3, 4 and 5.
    focus = hourly[hourly["day_index"].isin([3, 4, 5])]

    print("\nHourly details for days 3, 4 and 5:")
    print(focus.to_string())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    hourly.reset_index().to_csv(OUTPUT_PATH, index=False)

    print(f"\nHourly audit saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()