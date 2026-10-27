from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "processed" / "daily_activity.csv"
REPORT_DIR = ROOT / "data" / "reports"

TRAIN_DAYS = 18
VALIDATION_END = 24
SEASONAL_LAG = 7


def metrics(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def main():
    daily = pd.read_csv(INPUT_PATH).sort_values("day_index")
    daily = daily[daily["day_index"] <= 30].copy()
    daily = daily.reset_index(drop=True)

    y = daily["transaction_count"].to_numpy(dtype=float)
    days = daily["day_index"].to_numpy(dtype=int)

    if len(y) != 30:
        raise ValueError("Expected 30 complete daily observations.")

    if not np.isfinite(y).all():
        raise ValueError("Target contains non-finite values.")

    if TRAIN_DAYS <= SEASONAL_LAG:
        raise ValueError("Training history is too short.")

    records = []

    # One-step-ahead rolling forecasts.
    # At each origin, only earlier actual observations are used.
    for i in range(TRAIN_DAYS, len(y)):
        records.append({
            "day_index": int(days[i]),
            "actual": float(y[i]),
            "previous_day_forecast": float(y[i - 1]),
            "seasonal_naive_forecast": float(y[i - SEASONAL_LAG]),
            "split": (
                "validation" if i < VALIDATION_END else "test"
            ),
        })

    predictions = pd.DataFrame(records)

    results = {}

    for split in ["validation", "test"]:
        part = predictions[predictions["split"] == split]

        results[split] = {
            "previous_day": metrics(
                part["actual"], part["previous_day_forecast"]
            ),
            "seasonal_naive": metrics(
                part["actual"], part["seasonal_naive_forecast"]
            ),
        }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(
        REPORT_DIR / "rolling_baseline_predictions.csv",
        index=False,
    )

    report = {
        "target": "daily transaction count",
        "training_days": TRAIN_DAYS,
        "validation_days": VALIDATION_END - TRAIN_DAYS,
        "test_days": len(y) - VALIDATION_END,
        "evaluation": "one-step-ahead rolling forecast",
        "seasonal_lag_days": SEASONAL_LAG,
        "results": results,
        "limitation": (
            "Only 30 complete days are available. "
            "Weekly seasonality is not established."
        ),
    }

    (REPORT_DIR / "rolling_baseline_metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print("\nPredictions:")
    print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()