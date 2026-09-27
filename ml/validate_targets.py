from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/imd_rainfall_training_2024.csv"
)


def main():

    print("=" * 70)
    print("SIH26086 — TARGET VALIDATION")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"]
    )

    # -----------------------------------------------------
    # BASIC INFORMATION
    # -----------------------------------------------------

    print("\nDataset:")
    print(f"Rows       : {len(df):,}")
    print(
        f"Grid cells : "
        f"{df[['latitude', 'longitude']].drop_duplicates().shape[0]}"
    )

    # -----------------------------------------------------
    # OVERALL TARGET RATES
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("OVERALL TARGET RATES")
    print("=" * 70)

    onset_rate = df["onset_target_7d"].mean() * 100
    break_rate = df["break_target_7d"].mean() * 100

    print(
        f"Onset-positive observations : "
        f"{onset_rate:.2f}%"
    )

    print(
        f"Break-positive observations : "
        f"{break_rate:.2f}%"
    )

    # -----------------------------------------------------
    # MONTHLY DISTRIBUTION
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("MONTHLY TARGET DISTRIBUTION")
    print("=" * 70)

    df["month"] = df["date"].dt.month

    monthly = (
        df.groupby("month")
        .agg(
            observations=("date", "size"),
            onset_events=("onset_target_7d", "sum"),
            break_events=("break_target_7d", "sum")
        )
    )

    monthly["onset_percent"] = (
        monthly["onset_events"] /
        monthly["observations"] *
        100
    )

    monthly["break_percent"] = (
        monthly["break_events"] /
        monthly["observations"] *
        100
    )

    print(
        monthly.to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # -----------------------------------------------------
    # GRID-CELL DISTRIBUTION
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("GRID-CELL TARGET DISTRIBUTION")
    print("=" * 70)

    spatial = (
        df.groupby(
            ["latitude", "longitude"]
        )
        .agg(
            observations=("date", "size"),
            onset_rate=("onset_target_7d", "mean"),
            break_rate=("break_target_7d", "mean")
        )
        .reset_index()
    )

    print("\nOnset rate by grid cell:")

    print(
        spatial["onset_rate"]
        .describe()
        .to_string()
    )

    print("\nBreak rate by grid cell:")

    print(
        spatial["break_rate"]
        .describe()
        .to_string()
    )

    # -----------------------------------------------------
    # EXTREME GRID CELLS
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("GRID CELLS WITH VERY LOW/HIGH EVENT RATES")
    print("=" * 70)

    low_onset = spatial[
        spatial["onset_rate"] < 0.05
    ]

    high_onset = spatial[
        spatial["onset_rate"] > 0.95
    ]

    low_break = spatial[
        spatial["break_rate"] < 0.05
    ]

    high_break = spatial[
        spatial["break_rate"] > 0.95
    ]

    print(
        f"\nGrid cells with <5% onset events: "
        f"{len(low_onset)}"
    )

    print(
        f"Grid cells with >95% onset events: "
        f"{len(high_onset)}"
    )

    print(
        f"Grid cells with <5% break events: "
        f"{len(low_break)}"
    )

    print(
        f"Grid cells with >95% break events: "
        f"{len(high_break)}"
    )

    # -----------------------------------------------------
    # RAINFALL ASSOCIATION
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("RAINFALL ASSOCIATION WITH TARGETS")
    print("=" * 70)

    onset_rain = df.groupby(
        "onset_target_7d"
    )["rainfall_7d"].mean()

    break_rain = df.groupby(
        "break_target_7d"
    )["rainfall_7d"].mean()

    print("\nAverage previous 7-day rainfall by onset target:")

    print(onset_rain.to_string())

    print("\nAverage previous 7-day rainfall by break target:")

    print(break_rain.to_string())

    # -----------------------------------------------------
    # DATE RANGE
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("DATE COVERAGE")
    print("=" * 70)

    print(
        f"Start: {df['date'].min().date()}"
    )

    print(
        f"End  : {df['date'].max().date()}"
    )

    # -----------------------------------------------------
    # FINISH
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("TARGET VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()