
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "reports" / "rolling_model_comparison.csv"
OUTPUT_PATH = ROOT / "data" / "reports" / "model_error_analysis.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    # Har day ki actual count ko ek martaba rakhein.
    actuals = (
        df[["day_index", "split", "actual"]]
        .drop_duplicates()
    )

    # Har model ka absolute error alag columns mein laayein.
    errors = df.pivot(
        index="day_index",
        columns="model",
        values="absolute_error",
    ).reset_index()

    report = actuals.merge(errors, on="day_index")

    model_columns = [
        "previous_day",
        "training_median",
        "trailing_7_day_median",
    ]

    report["best_model_that_day"] = report[model_columns].idxmin(axis=1)
    report["largest_error_model"] = report[model_columns].idxmax(axis=1)

    print("\n=== DAY-BY-DAY ERROR ANALYSIS ===")
    print(
        report[
            [
                "day_index",
                "split",
                "actual",
                "previous_day",
                "training_median",
                "trailing_7_day_median",
                "best_model_that_day",
            ]
        ].to_string(index=False)
    )

    print("\n=== MODEL WINS BY SPLIT ===")
    for split in ["validation", "test"]:
        subset = report[report["split"] == split]
        print(f"\n{split.upper()}")

        print(
            subset["best_model_that_day"]
            .value_counts()
            .to_string()
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved analysis: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
