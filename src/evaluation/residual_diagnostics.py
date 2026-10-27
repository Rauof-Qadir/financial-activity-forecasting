
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

QUANTILES = [0.90, 0.95, 0.99]


def main():
    daily = pd.read_csv(INPUT_PATH).sort_values("day_index")
    daily = daily[daily["day_index"] <= MONITOR_END_DAY].copy()
    daily = daily.reset_index(drop=True)

    if daily["day_index"].tolist() != list(range(1, 31)):
        raise ValueError("Expected complete daily observations for days 1-30.")

    daily["forecast"] = daily["transaction_count"].shift(1)
    daily["residual"] = daily["transaction_count"] - daily["forecast"]
    daily["absolute_residual"] = daily["residual"].abs()

    training = daily[
        (daily["day_index"] >= 2)
        & (daily["day_index"] <= TRAIN_END_DAY)
    ].copy()

    monitored = daily[
        (daily["day_index"] >= MONITOR_START_DAY)
        & (daily["day_index"] <= MONITOR_END_DAY)
    ].copy()

    if len(training) < 10:
        raise ValueError("Too few training residuals for diagnostics.")

    # Candidate error thresholds come only from training residuals.
    thresholds = {
        str(int(q * 100)): float(
            training["absolute_residual"].quantile(q)
        )
        for q in QUANTILES
    }

    monitored["robust_z_score"] = (
        0.6745
        * (
            monitored["residual"]
            - training["residual"].median()
        )
        / training["residual"].mad()
        if hasattr(training["residual"], "mad")
        else np.nan
    )

    # Compute robust Z-scores explicitly using median absolute deviation.
    train_median = float(training["residual"].median())
    mad = float(np.median(np.abs(training["residual"] - train_median)))

    if mad > 0:
        monitored["robust_z_score"] = (
            0.6745 * (monitored["residual"] - train_median) / mad
        )
    else:
        monitored["robust_z_score"] = np.nan

    threshold_results = []

    for percentile, threshold in thresholds.items():
        mask = monitored["absolute_residual"] > threshold

        threshold_results.append({
            "training_abs_error_percentile": int(percentile),
            "threshold_transactions": round(threshold, 2),
            "monitoring_days_above_threshold": int(mask.sum()),
            "flagged_days": (
                monitored.loc[mask, "day_index"].astype(int).tolist()
            ),
        })

    output = monitored[
        [
            "day_index",
            "transaction_count",
            "forecast",
            "residual",
            "absolute_residual",
            "robust_z_score",
        ]
    ].copy()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = REPORT_DIR / "residual_diagnostics.csv"
    json_path = REPORT_DIR / "residual_diagnostics.json"

    output.to_csv(csv_path, index=False)

    summary = {
        "training_days": [2, TRAIN_END_DAY],
        "training_residual_count": int(len(training)),
        "monitoring_days": [MONITOR_START_DAY, MONITOR_END_DAY],
        "monitoring_residual_count": int(len(monitored)),
        "training_residual_median": train_median,
        "training_residual_mad": mad,
        "training_absolute_error_thresholds": thresholds,
        "threshold_comparison": threshold_results,
        "max_absolute_monitoring_residual": float(
            monitored["absolute_residual"].max()
        ),
        "max_absolute_robust_z_score": float(
            monitored["robust_z_score"].abs().max()
        ),
        "limitations": [
            "Only 17 training residuals and 12 monitoring residuals are available.",
            "Training quantiles are unstable with this small sample.",
            "Flagged volume changes are not confirmed fraud.",
            "This is exploratory analysis, not production threshold validation.",
        ],
    }

    json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print("\nMonitoring residuals:")
    print(output.to_string(index=False))
    print(f"\nCSV saved to: {csv_path}")
    print(f"JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
