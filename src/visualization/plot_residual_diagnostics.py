
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DAILY_PATH = ROOT / "data" / "processed" / "daily_activity.csv"
RESIDUAL_PATH = ROOT / "data" / "reports" / "residual_diagnostics.csv"
OUTPUT_PATH = ROOT / "data" / "reports" / "residual_diagnostics.png"


def main():
    daily = pd.read_csv(DAILY_PATH).sort_values("day_index")
    residuals = pd.read_csv(RESIDUAL_PATH).sort_values("day_index")

    daily = daily[daily["day_index"] <= 30]

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    # Chart 1: actual transaction activity
    axes[0].plot(
        daily["day_index"],
        daily["transaction_count"],
        marker="o",
        label="Actual daily transactions",
    )
    axes[0].axvline(
        18.5,
        linestyle="--",
        label="Training / monitoring boundary",
    )
    axes[0].set_title("Daily Transaction Activity")
    axes[0].set_ylabel("Transaction count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Chart 2: forecast residuals
    axes[1].bar(
        residuals["day_index"],
        residuals["residual"],
        label="Actual - forecast",
    )
    axes[1].axhline(0, linewidth=1)
    axes[1].axvline(
        18.5,
        linestyle="--",
        label="Training / monitoring boundary",
    )
    axes[1].set_title("One-Step-Ahead Forecast Errors")
    axes[1].set_xlabel("Day index")
    axes[1].set_ylabel("Residual (transactions)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved chart: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
