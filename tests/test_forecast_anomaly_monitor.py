
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "data" / "reports"

CSV_PATH = REPORT_DIR / "forecast_anomaly_alerts.csv"
JSON_PATH = REPORT_DIR / "forecast_anomaly_summary.json"


def load_outputs():
    assert CSV_PATH.exists(), "Run the monitoring script first."
    assert JSON_PATH.exists(), "Monitoring summary is missing."

    alerts = pd.read_csv(CSV_PATH)
    summary = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    return alerts, summary


def test_monitoring_covers_only_days_19_to_30():
    alerts, summary = load_outputs()

    assert alerts["day_index"].tolist() == list(range(19, 31))
    assert summary["monitoring_days"] == [19, 30]
    assert summary["monitored_days"] == 12


def test_alerts_match_the_configured_threshold():
    alerts, summary = load_outputs()

    expected = (
        alerts["robust_z_score"].abs()
        >= summary["robust_z_threshold"]
    )

    assert np.array_equal(
        alerts["alert"].to_numpy(dtype=bool),
        expected.to_numpy(dtype=bool),
    )


def test_monitoring_scores_are_finite():
    alerts, _ = load_outputs()

    assert alerts["robust_z_score"].notna().all()
    assert np.isfinite(
        alerts["robust_z_score"].to_numpy(dtype=float)
    ).all()


def test_summary_alert_count_matches_csv():
    alerts, summary = load_outputs()

    assert int(alerts["alert"].sum()) == summary["alert_count"]

    expected_days = (
        alerts.loc[alerts["alert"], "day_index"]
        .astype(int)
        .tolist()
    )

    assert expected_days == summary["alert_days"]


def test_alerts_do_not_claim_confirmed_fraud():
    alerts, _ = load_outputs()

    assert alerts["interpretation"].str.contains(
        "does not prove fraud", case=False
    ).all()
