from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "processed" / "daily_activity.csv"
OUTPUT_PATH = ROOT / "data" / "reports" / "daily_activity.png"


def main():
    daily = pd.read_csv(INPUT_PATH)

    # Day 31 has only 23 observed hours.
    daily = daily[daily["day_index"] <= 30].copy()

    if daily.empty:
        raise ValueError("No complete daily observations found.")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(
        daily["day_index"],
        daily["transaction_count"],
        marker="o",
    )
    axes[0].set_title("Daily Transaction Volume")
    axes[0].set_xlabel("Simulated day")
    axes[0].set_ylabel("Transaction count")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        daily["day_index"],
        daily["total_amount"] / 1_000_000_000,
        marker="o",
    )
    axes[1].set_title("Daily Transaction Amount")
    axes[1].set_xlabel("Simulated day")
    axes[1].set_ylabel("Total amount (billions)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Complete days:", len(daily))
    print("Plot saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()