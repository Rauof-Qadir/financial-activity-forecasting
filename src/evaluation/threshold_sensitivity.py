
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "reports" / "forecast_anomaly_alerts.csv"
OUTPUT_DIR = ROOT / "data" / "reports"

THRESHOLDS = [2.0, 2.5, 3.0, 3.5, 4.0]


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Run forecast_anomaly_monitor.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    if df.empty:
        raise ValueError("Monitoring report is empty.")

    if df["robust_z_score"].isna().any():
        raise ValueError("Robust Z-scores contain missing values.")

    scores = df["robust_z_score"].abs()
    results = []

    for threshold in THRESHOLDS:
        mask = scores >= threshold
        alert_days = df.loc[mask, "day_index"].astype(int).tolist()

        results.append({
            "threshold": threshold,
            "monitored_days": int(len(df)),
            "alert_count": int(mask.sum()),
            "alert_percentage": round(float(mask.mean() * 100), 2),
            "alert_days": alert_days,
        })

    result_df = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "threshold_sensitivity.csv"
    json_path = OUTPUT_DIR / "threshold_sensitivity.json"

    result_df.to_csv(csv_path, index=False)

    report = {
        "purpose": (
            "Compare alert counts under different robust Z-score thresholds."
        ),
        "monitored_period": [
            int(df["day_index"].min()),
            int(df["day_index"].max()),
        ],
        "threshold_results": results,
        "limitations": [
            "Only 12 monitoring days are available.",
            "Alert counts do not measure fraud detection performance.",
            "No threshold is validated for production use.",
        ],
    }

    json_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"\nCSV saved to: {csv_path}")
    print(f"JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
