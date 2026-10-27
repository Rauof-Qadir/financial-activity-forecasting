
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "data" / "reports" / "rolling_baseline_predictions.csv"
)
OUTPUT_PATH = (
    ROOT / "data" / "reports" / "baseline_comparison.png"
)


def main():
    df = pd.read_csv(INPUT_PATH).sort_values("day_index")

    if df.empty:
        raise ValueError("No predictions found.")

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 9), sharex=True
    )

    for ax, split in zip(axes, ["validation", "test"]):
        part = df[df["split"] == split]

        ax.plot(
            part["day_index"],
            part["actual"],
            marker="o",
            linewidth=2,
            label="Actual transactions",
        )
        ax.plot(
            part["day_index"],
            part["previous_day_forecast"],
            marker="s",
            linestyle="--",
            label="Previous-day baseline",
        )
        ax.plot(
            part["day_index"],
            part["seasonal_naive_forecast"],
            marker="^",
            linestyle=":",
            label="Seasonal-naive (lag 7)",
        )

        ax.set_title(f"{split.title()} period")
        ax.set_ylabel("Transaction count")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Day index")

    fig.suptitle(
        "Financial Activity Forecasting — Baseline Comparison"
    )
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Chart saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
