from pathlib import Path

import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data" / "processed" / "daily_activity.csv"
REPORT_DIR = ROOT / "data" / "reports"

TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20


def calculate_metrics(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
    }


def main():
    daily = pd.read_csv(INPUT_PATH).sort_values("day_index")

    # Use only complete days: 24 hourly steps per day.
    # In this dataset, day 31 is incomplete.
    daily = daily[daily["day_index"] <= 30].copy()

    if daily["day_index"].duplicated().any():
        raise ValueError("Duplicate daily observations found.")

    if len(daily) < 15:
        raise ValueError("Too few observations for this split.")

    values = daily["transaction_count"].to_numpy(dtype=float)
    days = daily["day_index"].to_numpy(dtype=int)

    if not np.isfinite(values).all():
        raise ValueError("Target contains non-finite values.")

    n = len(daily)
    train_end = int(n * TRAIN_FRACTION)
    validation_end = int(
        n * (TRAIN_FRACTION + VALIDATION_FRACTION)
    )

    train_y = values[:train_end]
    val_y = values[train_end:validation_end]
    test_y = values[validation_end:]

    val_days = days[train_end:validation_end]
    test_days = days[validation_end:]

    # Fixed-origin naive forecasts:
    # every forecast uses the last value from training only.
    last_train_value = float(train_y[-1])

    val_pred = np.full(len(val_y), last_train_value)
    test_pred = np.full(len(test_y), last_train_value)

    val_metrics = calculate_metrics(val_y, val_pred)
    test_metrics = calculate_metrics(test_y, test_pred)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    predictions = pd.DataFrame({
        "day_index": np.concatenate([val_days, test_days]),
        "actual_count": np.concatenate([val_y, test_y]),
        "predicted_count": np.concatenate([val_pred, test_pred]),
        "split": (
            ["validation"] * len(val_y)
            + ["test"] * len(test_y)
        ),
    })

    predictions["error"] = (
        predictions["actual_count"]
        - predictions["predicted_count"]
    )

    predictions.to_csv(
        REPORT_DIR / "naive_forecast_predictions.csv",
        index=False,
    )

    report = {
        "target": "daily transaction count",
        "complete_days": int(n),
        "train_days": int(len(train_y)),
        "validation_days": int(len(val_y)),
        "test_days": int(len(test_y)),
        "forecast_strategy": "fixed-origin naive",
        "last_training_value": last_train_value,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "limitation": (
            "Only 30 complete daily observations are available. "
            "Results are preliminary and may not generalize."
        ),
    }

    (REPORT_DIR / "naive_forecast_metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print("\nForecast predictions:")
    print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()