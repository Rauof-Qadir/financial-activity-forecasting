
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data" / "processed" / "daily_activity.csv"
REPORT_DIR = ROOT / "data" / "reports"

TRAIN_END_DAY = 18
MONITOR_START_DAY = 19
MONITOR_END_DAY = 30

ROBUST_Z_THRESHOLD = 3.5


def main():
    daily = pd.read_csv(INPUT_PATH).sort_values("day_index")
    daily = daily[daily["day_index"] <= MONITOR_END_DAY].copy()
    daily = daily.reset_index(drop=True)

    if daily["day_index"].tolist() != list(range(1, 31)):
        raise ValueError("Expected daily observations for days 1 through 30.")

    if daily["transaction_count"].isna().any():
        raise ValueError("Transaction counts contain missing values.")

    if (daily["transaction_count"] < 0).any():
        raise ValueError("Transaction counts cannot be negative.")

    # One-step-ahead forecast: yesterday's actual count.
    daily["forecast"] = daily["transaction_count"].shift(1)

    # Positive residual = more transactions than forecast.
    # Negative residual = fewer transactions than forecast.
    daily["residual"] = (
        daily["transaction_count"] - daily["forecast"]
    )

    # Learn the normal error pattern using training days only.
    training = daily[
        (daily["day_index"] >= 2)
        & (daily["day_index"] <= TRAIN_END_DAY)
    ].copy()

    train_residuals = training["residual"].to_numpy(dtype=float)

    residual_median = float(np.median(train_residuals))
    mad = float(np.median(np.abs(train_residuals - residual_median)))

    # MAD-based robust scale is less sensitive to extreme training errors.
    robust_scale = 1.4826 * mad

    if robust_scale == 0:
        raise ValueError(
            "Training residual variation is zero. "
            "Choose a different monitoring method or investigate the data."
        )

    daily["robust_z_score"] = (
        0.6745 * (daily["residual"] - residual_median) / mad
        if mad > 0
        else np.nan
    )

    # Monitor only after the training period.
    monitored = daily[
        (daily["day_index"] >= MONITOR_START_DAY)
        & (daily["day_index"] <= MONITOR_END_DAY)
    ].copy()

    monitored["alert"] = (
        monitored["robust_z_score"].abs() >= ROBUST_Z_THRESHOLD
    )

    monitored["alert_reason"] = np.where(
        monitored["alert"] & (monitored["residual"] > 0),
        "Unusually high transaction volume",
        np.where(
            monitored["alert"] & (monitored["residual"] < 0),
            "Unusually low transaction volume",
            "Within monitoring threshold",
        ),
    )


    monitored["interpretation"] = (
        "Volume monitoring is not a fraud determination; "
        "this result does not prove fraud or its absence."
    )


    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_csv = REPORT_DIR / "forecast_anomaly_alerts.csv"
    output_json = REPORT_DIR / "forecast_anomaly_summary.json"

    monitored.to_csv(output_csv, index=False)

    summary = {
        "target": "daily transaction count",
        "forecast_method": "previous day's actual transaction count",
        "training_days": [2, TRAIN_END_DAY],
        "monitoring_days": [MONITOR_START_DAY, MONITOR_END_DAY],
        "threshold_method": "robust z-score using training residuals",
        "robust_z_threshold": ROBUST_Z_THRESHOLD,
        "training_residual_median": residual_median,
        "training_residual_mad": mad,
        "training_robust_scale": robust_scale,
        "monitored_days": int(len(monitored)),
        "alert_count": int(monitored["alert"].sum()),
        "alert_days": monitored.loc[
            monitored["alert"], "day_index"
        ].astype(int).tolist(),
        "important_limitation": (
            "Only 18 training residuals and 12 monitoring days are available. "
            "Alerts indicate unusual volume, not confirmed fraud. "
            "Thresholds require more historical data and operational validation."
        ),
    }

    output_json.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print("\nMonitoring results:")
    print(
        monitored[
            [
                "day_index",
                "transaction_count",
                "forecast",
                "residual",
                "robust_z_score",
                "alert",
                "alert_reason",
            ]
        ].to_string(index=False)
    )

    print(f"\nCSV saved to: {output_csv}")
    print(f"Summary saved to: {output_json}")


if __name__ == "__main__":
    main()
