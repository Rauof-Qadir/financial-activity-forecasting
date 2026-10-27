from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "daily_activity.csv"
)

CHUNK_SIZE = 250_000
STEPS_PER_DAY = 24


def build_daily_series():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")

    step_summaries = []

    for chunk in pd.read_csv(
        RAW_PATH,
        usecols=["step", "amount"],
        chunksize=CHUNK_SIZE,
    ):
        chunk["step"] = pd.to_numeric(
            chunk["step"], errors="coerce"
        )
        chunk["amount"] = pd.to_numeric(
            chunk["amount"], errors="coerce"
        )

        if chunk[["step", "amount"]].isna().any().any():
            raise ValueError(
                "Missing or invalid step/amount values found. "
                "Review the data audit before continuing."
            )

        if not np.isfinite(chunk[["step", "amount"]].to_numpy()).all():
            raise ValueError("Non-finite numeric values found.")

        if not np.equal(
            chunk["step"].to_numpy(),
            np.floor(chunk["step"].to_numpy()),
        ).all():
            raise ValueError("Time steps must be whole numbers.")

        if (chunk["step"] < 1).any():
            raise ValueError("Expected time steps >= 1.")

        if (chunk["amount"] < 0).any():
            raise ValueError(
                "Negative amounts require investigation "
                "before aggregation."
            )

        grouped = chunk.groupby("step").agg(
            transaction_count=("amount", "size"),
            total_amount=("amount", "sum"),
        )

        step_summaries.append(grouped)

    if not step_summaries:
        raise ValueError("The input dataset is empty.")

    # Combine partial aggregates from all chunks.
    hourly = pd.concat(step_summaries).groupby(level=0).agg(
        transaction_count=("transaction_count", "sum"),
        total_amount=("total_amount", "sum"),
    )

    hourly.index = hourly.index.astype(int)
    hourly.index.name = "step"
    hourly = hourly.sort_index()

    # Do not silently treat missing hours as zero activity.
    expected_steps = np.arange(
        hourly.index.min(),
        hourly.index.max() + 1,
    )
    missing_steps = np.setdiff1d(expected_steps, hourly.index)

    if len(missing_steps):
        raise ValueError(
            f"Missing {len(missing_steps)} time steps. "
            f"Examples: {missing_steps[:10].tolist()}. "
            "Investigate before forecasting."
        )

    # Assumption: each step represents one simulated hour.
    hourly["day_index"] = (
        (hourly.index - 1) // STEPS_PER_DAY + 1
    )

    daily = hourly.groupby("day_index").agg(
        transaction_count=("transaction_count", "sum"),
        total_amount=("total_amount", "sum"),
    )

    daily["average_amount"] = (
        daily["total_amount"] / daily["transaction_count"]
    )

    daily.index.name = "day_index"
    daily = daily.reset_index()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUTPUT_PATH, index=False)

    print("\nDaily time-series preview:")
    print(daily.head(10).to_string(index=False))

    print("\nDataset summary:")
    print(f"Daily observations: {len(daily)}")
    print(f"Total transactions: {daily['transaction_count'].sum():,}")
    print(
        f"Total transaction amount: "
        f"{daily['total_amount'].sum():,.2f}"
    )
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_daily_series()