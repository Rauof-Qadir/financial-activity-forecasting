
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]

DAILY_PATH = ROOT / "data" / "processed" / "daily_activity.csv"
OUTPUT_PATH = ROOT / "data" / "reports" / "rolling_model_comparison.csv"

TRAIN_END = 18
VALIDATION_END = 24
LAST_COMPLETE_DAY = 30
ROLLING_WINDOW = 7


def calculate_metrics(actual, predicted):
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": np.sqrt(mean_squared_error(actual, predicted)),
    }


def main():
    df = pd.read_csv(DAILY_PATH).sort_values("day_index")

    # Day 31 incomplete hai, is liye exclude karein.
    df = df[df["day_index"] <= LAST_COMPLETE_DAY].copy()

    if df["day_index"].tolist() != list(range(1, 31)):
        raise ValueError("Expected complete daily observations for days 1–30.")

    counts = df.set_index("day_index")["transaction_count"]
    training_median = counts.loc[1:TRAIN_END].median()

    rows = []

    for day in range(TRAIN_END + 1, LAST_COMPLETE_DAY + 1):
        actual = counts.loc[day]

        # Baseline 1: previous day's actual count
        previous_day_forecast = counts.loc[day - 1]

        # Baseline 2: median learned from training data only
        median_forecast = training_median

        # Baseline 3: median of the previous 7 available days
        history = counts.loc[day - ROLLING_WINDOW:day - 1]
        trailing_median_forecast = history.median()

        split = "validation" if day <= VALIDATION_END else "test"

        forecasts = {
            "previous_day": previous_day_forecast,
            "training_median": median_forecast,
            "trailing_7_day_median": trailing_median_forecast,
        }

        for model, forecast in forecasts.items():
            rows.append({
                "day_index": day,
                "split": split,
                "model": model,
                "actual": int(actual),
                "forecast": float(forecast),
                "absolute_error": abs(actual - forecast),
                "squared_error": (actual - forecast) ** 2,
            })

    predictions = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_PATH, index=False)

    print("\n=== BASELINE COMPARISON ===")

    for split in ["validation", "test"]:
        print(f"\n--- {split.upper()} ---")
        subset = predictions[predictions["split"] == split]

        for model in forecasts:
            model_rows = subset[subset["model"] == model]
            metrics = calculate_metrics(
                model_rows["actual"],
                model_rows["forecast"],
            )

            print(
                f"{model:24s} "
                f"MAE={metrics['mae']:,.2f} "
                f"RMSE={metrics['rmse']:,.2f}"
            )

    print(f"\nSaved predictions: {OUTPUT_PATH}")
    print(
        "\nModel selection should be based on validation results. "
        "Test results are reported separately for transparency."
    )


if __name__ == "__main__":
    main()
