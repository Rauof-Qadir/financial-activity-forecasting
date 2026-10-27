
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

DAILY_PATH = ROOT / "data" / "processed" / "daily_activity.csv"
COMPARISON_PATH = ROOT / "data" / "reports" / "rolling_model_comparison.csv"
ERROR_PATH = ROOT / "data" / "reports" / "model_error_analysis.csv"
ALERT_PATH = ROOT / "data" / "reports" / "forecast_anomaly_alerts.csv"

st.set_page_config(
    page_title="FinPulse | Risk Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# THEME
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #0B1120;
        --panel: #111B2E;
        --border: #25344B;
        --text: #E8EEF8;
        --muted: #91A2BB;
        --teal: #45E0C1;
        --blue: #77A9FF;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 17, 32, 0.92);
    }

    [data-testid="stSidebar"] {
        background: #0E1728;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.5px;
        color: var(--text);
    }

    p, label, [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1600px;
    }

    .brand {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -1px;
    }

    .brand span {
        color: var(--teal);
    }

    .eyebrow {
        color: var(--teal);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }

    .hero {
        padding: 1.8rem 2rem;
        border: 1px solid var(--border);
        border-radius: 20px;
        background:
            radial-gradient(ellipse at 90% 0%, #173D4B 0%, transparent 48%),
            linear-gradient(135deg, #131F33 0%, #0F192A 100%);
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        font-size: clamp(1.7rem, 3vw, 2.65rem);
        margin: 0 0 0.65rem 0;
        line-height: 1.15;
    }

    .hero p {
        font-size: 0.96rem;
        max-width: 750px;
        line-height: 1.7;
        margin: 0;
    }

    .section-heading {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.28rem;
        font-weight: 600;
        margin: 1.3rem 0 0.25rem 0;
        color: var(--text);
    }

    .section-subtitle {
        font-size: 0.84rem;
        color: var(--muted);
        margin-bottom: 1rem;
    }

    .kpi {
        background: linear-gradient(145deg, #152239, #101A2C);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.15rem 1.2rem;
        min-height: 135px;
    }

    .kpi-label {
        font-size: 0.78rem;
        color: var(--muted);
        margin-bottom: 0.8rem;
    }

    .kpi-value {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text);
        font-size: clamp(1.35rem, 2vw, 1.9rem);
        font-weight: 700;
        line-height: 1.2;
        overflow-wrap: anywhere;
    }

    .kpi-note {
        font-size: 0.74rem;
        color: var(--teal);
        margin-top: 0.65rem;
    }

    .panel {
        background: #111B2E;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.1rem 1.2rem;
    }

    .status-pill {
        display: inline-block;
        padding: 5px 10px;
        border: 1px solid #2C756D;
        border-radius: 999px;
        color: var(--teal);
        font-size: 0.72rem;
        font-weight: 600;
        background: #102D30;
    }

    .muted-small {
        font-size: 0.78rem;
        color: var(--muted);
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    .footer {
        border-top: 1px solid var(--border);
        margin-top: 2.5rem;
        padding-top: 1.2rem;
        font-size: 0.78rem;
        color: var(--muted);
        line-height: 1.8;
    }

    hr {
        border-color: var(--border) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame | None:
    """Read a CSV safely. Return None if it is unavailable."""
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Could not read `{path.name}`: {exc}")
        return None


def section_title(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="section-heading">{title}</div>'
        f'<div class="section-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_theme(chart: alt.Chart) -> alt.Chart:
    return chart.configure_view(
        strokeOpacity=0
    ).configure_axis(
        labelColor="#91A2BB",
        titleColor="#91A2BB",
        gridColor="#25344B",
        domainColor="#25344B",
        tickColor="#25344B",
        labelFont="DM Sans",
        titleFont="DM Sans",
        labelFontSize=11,
        titleFontSize=12,
    ).configure_legend(
        labelColor="#C7D4E7",
        titleColor="#C7D4E7",
        labelFont="DM Sans",
        titleFont="DM Sans",
    ).configure_title(
        color="#E8EEF8",
        font="Space Grotesk",
    )


def make_line_chart(
    frame: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    tooltip: list[str] | None = None,
) -> alt.Chart:
    encoding = {
        "x": alt.X(
            f"{x}:Q",
            title="Day",
            axis=alt.Axis(tickMinStep=1),
        ),
        "y": alt.Y(f"{y}:Q", title=None),
        "tooltip": tooltip or [x, y],
    }

    if color:
        encoding["color"] = alt.Color(
            f"{color}:N",
            title="Model",
            scale=alt.Scale(
                range=["#45E0C1", "#77A9FF", "#F4B860", "#D79AFF"]
            ),
        )

    chart = (
        alt.Chart(frame)
        .mark_line(point=alt.OverlayMarkDef(size=65), strokeWidth=2.5)
        .encode(**encoding)
        .properties(height=320)
    )

    return chart_theme(chart)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="brand">Fin<span>Pulse</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("FINANCIAL INTELLIGENCE PLATFORM")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Executive Overview",
            "Activity Analytics",
            "Model Performance",
            "Risk Monitoring",
            "Data Explorer",
        ],
        label_visibility="visible",
    )

    st.divider()
    st.markdown("**PROJECT STATUS**")
    st.markdown(
        '<span class="status-pill">SIMULATED DATA</span>',
        unsafe_allow_html=True,
    )
    st.caption("Forecasting · Model evaluation · Volume monitoring")
    st.divider()
    st.caption("FINPULSE / VERSION 1.0")


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

daily = load_csv(DAILY_PATH)
comparison = load_csv(COMPARISON_PATH)
errors = load_csv(ERROR_PATH)

# Alert outputs may not have been generated yet.
alerts = load_csv(ALERT_PATH)

if daily is None or comparison is None:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">SYSTEM INITIALIZATION</div>
            <h1>Your analytics workspace</h1>
            <p>Some required project outputs are missing. Generate the
            daily activity and model comparison files, then refresh this app.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    missing = []
    if daily is None:
        missing.append(str(DAILY_PATH.relative_to(ROOT)))
    if comparison is None:
        missing.append(str(COMPARISON_PATH.relative_to(ROOT)))

    for path in missing:
        st.code(path)

    st.code(
        "python src/data/build_daily_series.py\n"
        "python src/evaluation/compare_rolling_models.py"
    )
    st.stop()

# Validate required columns before using them.
required_daily = {"day_index", "transaction_count"}
required_comparison = {"day_index", "split", "model", "actual", "forecast"}

if not required_daily.issubset(daily.columns):
    st.error(f"Daily data must contain: {sorted(required_daily)}")
    st.stop()

if not required_comparison.issubset(comparison.columns):
    st.error(f"Comparison data must contain: {sorted(required_comparison)}")
    st.stop()

daily = daily.copy()
daily["day_index"] = pd.to_numeric(daily["day_index"], errors="coerce")
daily["transaction_count"] = pd.to_numeric(
    daily["transaction_count"], errors="coerce"
)
daily = daily.dropna(subset=["day_index", "transaction_count"])
daily = daily[daily["day_index"] <= 30].sort_values("day_index")

comparison = comparison.copy()
for column in ["day_index", "actual", "forecast"]:
    comparison[column] = pd.to_numeric(
        comparison[column], errors="coerce"
    )
comparison = comparison.dropna(
    subset=["day_index", "actual", "forecast", "split", "model"]
)

if daily.empty or comparison.empty:
    st.error("No valid daily observations or model predictions were found.")
    st.stop()

# --------------------------------------------------
# SHARED METRICS
# --------------------------------------------------

total_transactions = int(daily["transaction_count"].sum())
average_daily = float(daily["transaction_count"].mean())
peak_day = daily.loc[daily["transaction_count"].idxmax()]
lowest_day = daily.loc[daily["transaction_count"].idxmin()]

comparison["absolute_error"] = (
    comparison["actual"] - comparison["forecast"]
).abs()
comparison["squared_error"] = (
    comparison["actual"] - comparison["forecast"]
) ** 2

metrics = (
    comparison.groupby(["split", "model"], as_index=False)
    .agg(
        MAE=("absolute_error", "mean"),
        MSE=("squared_error", "mean"),
    )
)
metrics["RMSE"] = metrics["MSE"] ** 0.5
metrics = metrics.drop(columns=["MSE"])

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">FINANCIAL DATA / RISK INTELLIGENCE</div>
        <h1>Financial Activity<br>Forecasting &amp; Monitoring</h1>
        <p>Understand transaction-volume patterns, compare forecasting
        strategies, and identify unusual changes in simulated financial activity.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if page == "Executive Overview":
    st.markdown("### Executive overview")
    st.caption("A high-level snapshot of activity and forecast performance.")

    cols = st.columns(4)
    with cols[0]:
        kpi_card("COMPLETE DAYS", f"{len(daily)}", "Full days included")
    with cols[1]:
        kpi_card("TOTAL TRANSACTIONS", f"{total_transactions:,}", "Across included days")
    with cols[2]:
        kpi_card("DAILY AVERAGE", f"{average_daily:,.0f}", "Transactions per day")
    with cols[3]:
        kpi_card(
            "PEAK ACTIVITY",
            f"Day {int(peak_day['day_index'])}",
            f"{int(peak_day['transaction_count']):,} transactions",
        )

    st.divider()

    left, right = st.columns([1.6, 1])

    with left:
        section_title(
            "Daily activity",
            "Transaction counts across the 30 complete simulated days.",
        )
        chart = (
            alt.Chart(daily)
            .mark_area(
                line={"color": "#45E0C1", "strokeWidth": 2.5},
                color=alt.Gradient(
                    gradient="linear",
                    stops=[
                        alt.GradientStop(color="#45E0C1", offset=0),
                        alt.GradientStop(color="#142E3A", offset=1),
                    ],
                    x1=1, x2=1, y1=0, y2=1,
                ),
                opacity=0.75,
            )
            .encode(
                x=alt.X("day_index:Q", title="Day", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("transaction_count:Q", title="Transactions"),
                tooltip=[
                    alt.Tooltip("day_index:Q", title="Day", format=".0f"),
                    alt.Tooltip("transaction_count:Q", title="Transactions", format=","),
                ],
            )
            .properties(height=310)
        )
        st.altair_chart(chart_theme(chart), use_container_width=True)

    with right:
        section_title("Activity summary", "Highest and lowest observed volume.")
        st.markdown(
            f"""
            <div class="panel">
                <div class="muted-small">HIGHEST DAILY VOLUME</div>
                <h2>{int(peak_day['transaction_count']):,}</h2>
                <p>Day {int(peak_day['day_index'])}</p>
                <hr>
                <div class="muted-small">LOWEST DAILY VOLUME</div>
                <h2>{int(lowest_day['transaction_count']):,}</h2>
                <p>Day {int(lowest_day['day_index'])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.info(
            "Volume changes are descriptive signals. They do not, by "
            "themselves, establish fraud or a business cause."
        )

    section_title("Model performance snapshot", "MAE by evaluation period and forecasting strategy.")

    available_splits = [
        s for s in ["validation", "test"] if s in metrics["split"].unique()
    ]
    selected = st.selectbox(
        "Evaluation period",
        available_splits or ["validation"],
        key="overview_split",
    )
    subset = metrics[metrics["split"] == selected].sort_values("MAE")

    if not subset.empty:
        bars = (
            alt.Chart(subset)
            .mark_bar(cornerRadiusEnd=6)
            .encode(
                y=alt.Y("model:N", sort="-x", title=None),
                x=alt.X("MAE:Q", title="Mean absolute error"),
                color=alt.Color(
                    "model:N",
                    legend=None,
                    scale=alt.Scale(
                        range=["#45E0C1", "#77A9FF", "#F4B860", "#D79AFF"]
                    ),
                ),
                tooltip=["model", alt.Tooltip("MAE:Q", format=",.2f"), alt.Tooltip("RMSE:Q", format=",.2f")],
            )
            .properties(height=190)
        )
        st.altair_chart(chart_theme(bars), use_container_width=True)
        st.dataframe(
            subset[["split", "model", "MAE", "RMSE"]].round(2),
            use_container_width=True,
            hide_index=True,
        )

elif page == "Activity Analytics":
    section_title(
        "Activity analytics",
        "Explore daily transaction volume and changes across the observed period.",
    )

    min_day = int(daily["day_index"].min())
    max_day = int(daily["day_index"].max())

    day_range = st.slider(
        "Select day range",
        min_value=min_day,
        max_value=max_day,
        value=(min_day, max_day),
    )
    selected_daily = daily[
        daily["day_index"].between(day_range[0], day_range[1])
    ].copy()

    st.columns(3)[0].metric("Days selected", len(selected_daily))
    st.columns(3)[1].metric(
        "Transactions in range",
        f"{int(selected_daily['transaction_count'].sum()):,}",
    )
    st.columns(3)[2].metric(
        "Average volume",
        f"{selected_daily['transaction_count'].mean():,.0f}",
    )

    line = (
        alt.Chart(selected_daily)
        .mark_line(
            point=alt.OverlayMarkDef(size=70, filled=True),
            color="#45E0C1",
            strokeWidth=3,
        )
        .encode(
            x=alt.X("day_index:Q", title="Day", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("transaction_count:Q", title="Transactions"),
            tooltip=[
                alt.Tooltip("day_index:Q", title="Day"),
                alt.Tooltip("transaction_count:Q", title="Transactions", format=","),
            ],
        )
        .properties(height=380)
    )
    st.altair_chart(chart_theme(line), use_container_width=True)

    st.dataframe(
        selected_daily.round(2),
        use_container_width=True,
        hide_index=True,
    )

elif page == "Model Performance":
    section_title(
        "Forecast model performance",
        "Compare one-step-ahead baselines on validation and test periods.",
    )

    st.info(
        "Choose the model using validation results. Test results are "
        "reported separately and should not be used to tune the model."
    )

    available_splits = [
        s for s in ["validation", "test"] if s in metrics["split"].unique()
    ]
    chosen_split = st.segmented_control(
        "Evaluation period",
        options=available_splits or ["validation"],
        default=(available_splits or ["validation"])[0],
    )

    selected_metrics = metrics[metrics["split"] == chosen_split].copy()

    if not selected_metrics.empty:
        best = selected_metrics.loc[selected_metrics["MAE"].idxmin()]
        c1, c2, c3 = st.columns(3)
        c1.metric("Best model in this period", str(best["model"]))
        c2.metric("Lowest MAE", f"{best['MAE']:,.2f}")
        c3.metric("RMSE", f"{best['RMSE']:,.2f}")

        bars = (
            alt.Chart(selected_metrics)
            .mark_bar(cornerRadiusEnd=5)
            .encode(
                x=alt.X("model:N", sort="-y", title=None),
                y=alt.Y("MAE:Q", title="Mean absolute error"),
                color=alt.Color(
                    "model:N",
                    legend=None,
                    scale=alt.Scale(
                        range=["#45E0C1", "#77A9FF", "#F4B860", "#D79AFF"]
                    ),
                ),
                tooltip=["model", alt.Tooltip("MAE:Q", format=",.2f"), alt.Tooltip("RMSE:Q", format=",.2f")],
            )
            .properties(height=320)
        )
        st.altair_chart(chart_theme(bars), use_container_width=True)

    st.markdown("#### Metric table")
    st.dataframe(
        metrics.sort_values(["split", "MAE"]).round(2),
        use_container_width=True,
        hide_index=True,
    )

    section_title(
        "Day-by-day forecast errors",
        "Compare absolute errors for each model on each evaluation day.",
    )

    split_options = [
        s for s in ["validation", "test"] if s in comparison["split"].unique()
    ]
    split = st.selectbox(
        "Choose period for error analysis",
        split_options or ["validation"],
        key="error_split",
    )
    period = comparison[comparison["split"] == split].copy()

    if not period.empty:
        error_chart = (
            alt.Chart(period)
            .mark_line(point=True, strokeWidth=2.5)
            .encode(
                x=alt.X("day_index:Q", title="Day", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("absolute_error:Q", title="Absolute forecast error"),
                color=alt.Color(
                    "model:N",
                    scale=alt.Scale(
                        range=["#45E0C1", "#77A9FF", "#F4B860", "#D79AFF"]
                    ),
                ),
                tooltip=[
                    "day_index",
                    "model",
                    alt.Tooltip("actual:Q", format=","),
                    alt.Tooltip("forecast:Q", format=","),
                    alt.Tooltip("absolute_error:Q", format=",.2f"),
                ],
            )
            .properties(height=360)
        )
        st.altair_chart(chart_theme(error_chart), use_container_width=True)

        st.dataframe(
            period[
                ["day_index", "model", "actual", "forecast", "absolute_error"]
            ].sort_values(["day_index", "model"]).round(2),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Open additional model error analysis"):
        if errors is not None:
            if "split" in errors.columns:
                st.dataframe(
                    errors[errors["split"] == split],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.dataframe(errors, use_container_width=True, hide_index=True)
        else:
            st.caption(
                "Optional error analysis file not found. Generate "
                "src/evaluation/analyze_model_errors.py first."
            )

elif page == "Risk Monitoring":
    section_title(
        "Forecast anomaly monitoring",
        "Review unusually high or low transaction-volume observations.",
    )

    if alerts is None:
        st.warning(
            "Monitoring output is not available yet. Run the monitoring "
            "script after generating daily activity data."
        )
        st.code("python src/monitoring/forecast_anomaly_monitor.py")
    else:
        required_alert_cols = {
            "day_index",
            "robust_z_score",
            "alert",
            "interpretation",
        }

        if not required_alert_cols.issubset(alerts.columns):
            st.error(
                "Monitoring CSV does not contain the expected columns: "
                f"{sorted(required_alert_cols)}"
            )
        else:
            alerts = alerts.copy()
            alerts["alert"] = (
                alerts["alert"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["true", "1", "yes"])
            )
            alerts["robust_z_score"] = pd.to_numeric(
                alerts["robust_z_score"], errors="coerce"
            )

            flagged = alerts[alerts["alert"]]
            m1, m2, m3 = st.columns(3)
            m1.metric("Monitored days", len(alerts))
            m2.metric("Flagged observations", len(flagged))
            m3.metric(
                "Configured threshold",
                (
                    f"±{float(alerts['robust_z_threshold'].iloc[0]):.1f}"
                    if "robust_z_threshold" in alerts.columns and not alerts.empty
                    else "See summary"
                ),
            )

            if flagged.empty:
                st.success(
                    "No observations crossed the configured threshold "
                    "during this monitoring period."
                )
            else:
                st.warning(
                    f"{len(flagged)} observation(s) crossed the configured "
                    "volume threshold. Review them as investigation leads."
                )

            plot_data = alerts.dropna(subset=["robust_z_score"])
            if not plot_data.empty:
                anomaly_chart = (
                    alt.Chart(plot_data)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X("day_index:Q", title="Day", axis=alt.Axis(tickMinStep=1)),
                        y=alt.Y("robust_z_score:Q", title="Robust z-score"),
                        color=alt.condition(
                            "datum.alert == true",
                            alt.value("#F4B860"),
                            alt.value("#45E0C1"),
                        ),
                        tooltip=[
                            "day_index",
                            alt.Tooltip("robust_z_score:Q", format=".3f"),
                            "alert",
                        ],
                    )
                    .properties(height=320)
                )
                st.altair_chart(chart_theme(anomaly_chart), use_container_width=True)

            st.markdown("#### Monitoring records")
            st.dataframe(alerts, use_container_width=True, hide_index=True)

            st.warning(
                "A volume alert is not a fraud prediction. No alert does "
                "not establish the absence of fraud. Investigate alerts "
                "with transaction-level evidence and validated controls."
            )

elif page == "Data Explorer":
    section_title(
        "Data explorer",
        "Inspect the daily series and the model evaluation records.",
    )

    dataset_choice = st.selectbox(
        "Choose dataset",
        ["Daily activity", "Model predictions", "Monitoring alerts"],
    )

    if dataset_choice == "Daily activity":
        st.dataframe(daily, use_container_width=True, hide_index=True)
        st.download_button(
            "Download daily activity CSV",
            daily.to_csv(index=False).encode("utf-8"),
            file_name="daily_activity.csv",
            mime="text/csv",
        )

    elif dataset_choice == "Model predictions":
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.download_button(
            "Download model predictions CSV",
            comparison.to_csv(index=False).encode("utf-8"),
            file_name="rolling_model_comparison.csv",
            mime="text/csv",
        )

    else:
        if alerts is None:
            st.info("Monitoring alerts are not available yet.")
        else:
            st.dataframe(alerts, use_container_width=True, hide_index=True)
            st.download_button(
                "Download monitoring CSV",
                alerts.to_csv(index=False).encode("utf-8"),
                file_name="forecast_anomaly_alerts.csv",
                mime="text/csv",
            )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        <strong>FINPULSE</strong> &nbsp; · &nbsp; Financial Activity Forecasting &amp; Risk Monitoring
        <br>
        Built for exploratory analytics, baseline evaluation and volume monitoring.
        <br>
        <span>Data note: This project uses simulated transaction data.
        Findings are preliminary and are not proof of real-world financial behavior or fraud.</span>
    </div>
    """,
    unsafe_allow_html=True,
)
