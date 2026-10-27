
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DAILY_PATH = ROOT / "data" / "processed" / "daily_activity.csv"
PREDICTIONS_PATH = (
    ROOT / "data" / "reports" / "rolling_baseline_predictions.csv"
)


def load_data():
    daily = pd.read_csv(DAILY_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    return daily, predictions


def test_evaluation_uses_only_complete_days():
    daily, predictions = load_data()

    assert daily["day_index"].max() == 31
    assert predictions["day_index"].max() == 30
    assert 31 not in predictions["day_index"].values


def test_validation_and_test_are_chronological():
    _, predictions = load_data()

    validation = predictions[predictions["split"] == "validation"]
    test = predictions[predictions["split"] == "test"]

    assert validation["day_index"].tolist() == [19, 20, 21, 22, 23, 24]
    assert test["day_index"].tolist() == [25, 26, 27, 28, 29, 30]
    assert validation["day_index"].max() < test["day_index"].min()


def test_previous_day_forecast_uses_previous_actual():
    daily, predictions = load_data()

    actual_by_day = daily.set_index("day_index")["transaction_count"]

    for row in predictions.itertuples():
        expected = actual_by_day.loc[row.day_index - 1]
        assert np.isclose(row.previous_day_forecast, expected)


def test_seasonal_forecast_uses_value_from_seven_days_earlier():
    daily, predictions = load_data()

    actual_by_day = daily.set_index("day_index")["transaction_count"]

    for row in predictions.itertuples():
        expected = actual_by_day.loc[row.day_index - 7]
        assert np.isclose(row.seasonal_naive_forecast, expected)


def test_predictions_have_no_missing_or_invalid_values():
    _, predictions = load_data()

    numeric_columns = [
        "day_index",
        "actual",
        "previous_day_forecast",
        "seasonal_naive_forecast",
    ]

    assert predictions[numeric_columns].notna().all().all()
    assert np.isfinite(predictions[numeric_columns].to_numpy()).all()
    assert (predictions["actual"] >= 0).all()
