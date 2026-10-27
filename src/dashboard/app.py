
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

DAILY_PATH = ROOT / "data" / "processed" / "daily_activity.csv"
COMPARISON_PATH = ROOT / "data" / "reports" / "rolling_model_comparison.csv"
ERROR_PATH = ROOT / "data" / "reports" / "model_error_analysis.csv"
ALERT_PATH = ROOT / "data" / "reports" / "forecast_anomaly_alerts.csv"

st.set_page_config(
    page_title="Financial Activity Forecasting",
    page_icon="📈",
    layout="wide",
)

st.title("Financial Activity Forecasting & Risk Monitoring")
st.caption(
    "Daily transaction-volume analytics, baseline evaluation, "
    "and forecast monitoring."
)

# Load project outputs
daily = pd.read_csv(DAILY_PATH)
daily = daily[daily["day_index"] <= 30].sort_values("day_index")

comparison = pd.read_csv(COMPARISON_PATH)
errors = pd.read_csv(ERROR_PATH)

total_transactions = int(daily["transaction_count"].sum())
average_daily = daily["transaction_count"].mean()
peak_day = daily.loc[
    daily["transaction_count"].idxmax()
]
lowest_day = daily.loc[
    daily["transaction_count"].idxmin()
]

# KPI cards
c1, c2, c3, c4 = st.columns(4)

c1.metric("Complete Days", len(daily))
c2.metric("Transactions", f"{total_transactions:,}")
c3.metric("Average Daily Volume", f"{average_daily:,.0f}")
c4.metric("Highest Volume Day", f"Day {int(peak_day['day_index'])}")

st.divider()

# Daily activity
st.subheader("1. Daily Transaction Volume")
st.line_chart(
    daily.set_index("day_index")[["transaction_count"]]
)
st.caption(
    "Day 31 is excluded because it does not contain a complete 24-hour day."
)

# Model comparison
st.subheader("2. Forecast Model Comparison")

metric_rows = []

for split in ["validation", "test"]:
    subset = comparison[comparison["split"] == split]

    for model, group in subset.groupby("model"):
        mae = (group["actual"] - group["forecast"]).abs().mean()
        rmse = (
            ((group["actual"] - group["forecast"]) ** 2).mean()
            ** 0.5
        )

        metric_rows.append({
            "split": split,
            "model": model,
            "MAE": mae,
            "RMSE": rmse,
        })

metrics = pd.DataFrame(metric_rows)

left, right = st.columns(2)

with left:
    st.markdown("**Validation MAE**")
    validation = metrics[metrics["split"] == "validation"]
    st.bar_chart(validation.set_index("model")[["MAE"]])

with right:
    st.markdown("**Test MAE**")
    test = metrics[metrics["split"] == "test"]
    st.bar_chart(test.set_index("model")[["MAE"]])

st.dataframe(
    metrics.sort_values(["split", "MAE"]),
    use_container_width=True,
    hide_index=True,
)

st.info(
    "Model-selection decision: previous-day is the selected baseline "
    "because it achieved the lowest validation MAE. Test results are "
    "reported separately; they do not determine the selection."
)

# Daily errors
st.subheader("3. Day-by-Day Forecast Errors")

selected_split = st.selectbox(
    "Choose evaluation period",
    ["validation", "test"],
)

period = comparison[comparison["split"] == selected_split].copy()
period["absolute_error"] = (
    period["actual"] - period["forecast"]
).abs()

error_pivot = period.pivot(
    index="day_index",
    columns="model",
    values="absolute_error",
)

st.line_chart(error_pivot)
st.dataframe(
    errors[errors["split"] == selected_split],
    use_container_width=True,
    hide_index=True,
)

# Monitoring results
st.subheader("4. Forecast Anomaly Monitoring")

if ALERT_PATH.exists():
    alerts = pd.read_csv(ALERT_PATH)

    st.write(
        "Monitoring output from the project's forecast anomaly detector."
    )
    st.dataframe(alerts, use_container_width=True, hide_index=True)

    alert_columns = [
        column for column in alerts.columns
        if column.lower() in {"alert", "is_alert", "anomaly_alert"}
    ]

    if alert_columns:
        alert_count = int(alerts[alert_columns[0]].astype(bool).sum())
        st.metric("Flagged Monitoring Rows", alert_count)
else:
    st.warning(
        "Monitoring output was not found. Run the anomaly-monitoring "
        "script before using this section."
    )

st.warning(
    "Forecast-volume alerts are investigation signals, not fraud labels. "
    "No alert does not prove that no anomaly or fraud occurred."
)

st.divider()
st.caption(
    "Data note: This project uses simulated transaction data. "
    "Observed patterns should not be treated as proven real-world "
    "financial behavior."
)
