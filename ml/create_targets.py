from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/imd_rainfall_features_2024.csv"
)

OUTPUT_FILE = Path(
    "data/processed/imd_rainfall_training_2024.csv"
)


# ---------------------------------------------------------
# DEVELOPMENT THRESHOLDS
# ---------------------------------------------------------

# A day is considered rainy when rainfall exceeds this.
RAIN_THRESHOLD = 2.5

# Minimum number of rainy days in a 7-day future window
# used for the development onset indicator.
ONSET_RAINY_DAYS = 4

# Minimum dry days in a future 7-day window used for the
# development break indicator.
BREAK_DRY_DAYS = 5


def main():

    print("=" * 70)
    print("SIH26086 — TARGET ENGINEERING")
    print("=" * 70)

    # -----------------------------------------------------
    # LOAD
    # -----------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    print("\nLoading feature dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"]
    )

    df = df.sort_values(
        ["latitude", "longitude", "date"]
    ).reset_index(drop=True)

    print(
        f"Input rows: {len(df):,}"
    )

    # -----------------------------------------------------
    # FUTURE RAINFALL
    # -----------------------------------------------------

    grouped = df.groupby(
        ["latitude", "longitude"],
        sort=False
    )

    print(
        "\nCreating future rainfall targets..."
    )

    # Future 7-day rainfall total.
    future_7d_values = (
        grouped["rainfall_mm"]
        .transform(
            lambda x: (
                x.shift(-1)
                .rolling(
                    window=7,
                    min_periods=7
                )
                .sum()
                .shift(-6)
            )
        )
    )

    df["future_rainfall_7d"] = future_7d_values

    # Future 14-day rainfall total.
    future_14d_values = (
        grouped["rainfall_mm"]
        .transform(
            lambda x: (
                x.shift(-1)
                .rolling(
                    window=14,
                    min_periods=14
                )
                .sum()
                .shift(-13)
            )
        )
    )

    df["future_rainfall_14d"] = future_14d_values

    # Future 30-day rainfall total.
    future_30d_values = (
        grouped["rainfall_mm"]
        .transform(
            lambda x: (
                x.shift(-1)
                .rolling(
                    window=30,
                    min_periods=30
                )
                .sum()
                .shift(-29)
            )
        )
    )

    df["future_rainfall_30d"] = future_30d_values

    # -----------------------------------------------------
    # FUTURE RAINY DAYS
    # -----------------------------------------------------

    df["_future_rain"] = (
        df["rainfall_mm"] > RAIN_THRESHOLD
    ).astype(int)

    rainy_grouped = df.groupby(
        ["latitude", "longitude"],
        sort=False
    )

    df["future_rainy_days_7d"] = (
        rainy_grouped["_future_rain"]
        .transform(
            lambda x: (
                x.shift(-1)
                .rolling(
                    window=7,
                    min_periods=7
                )
                .sum()
                .shift(-6)
            )
        )
    )

    # -----------------------------------------------------
    # FUTURE DRY DAYS
    # -----------------------------------------------------

    df["_future_dry"] = (
        df["rainfall_mm"] <= RAIN_THRESHOLD
    ).astype(int)

    dry_grouped = df.groupby(
        ["latitude", "longitude"],
        sort=False
    )

    df["future_dry_days_7d"] = (
        dry_grouped["_future_dry"]
        .transform(
            lambda x: (
                x.shift(-1)
                .rolling(
                    window=7,
                    min_periods=7
                )
                .sum()
                .shift(-6)
            )
        )
    )

    # -----------------------------------------------------
    # ONSET TARGET
    # -----------------------------------------------------

    df["onset_target_7d"] = (
        df["future_rainy_days_7d"]
        >= ONSET_RAINY_DAYS
    ).astype(int)

    # -----------------------------------------------------
    # BREAK TARGET
    # -----------------------------------------------------

    df["break_target_7d"] = (
        df["future_dry_days_7d"]
        >= BREAK_DRY_DAYS
    ).astype(int)

    # -----------------------------------------------------
    # REMOVE TEMPORARY COLUMNS
    # -----------------------------------------------------

    df = df.drop(
        columns=[
            "_future_rain",
            "_future_dry"
        ]
    )

    # -----------------------------------------------------
    # REMOVE ROWS WITHOUT COMPLETE FUTURE WINDOW
    # -----------------------------------------------------

    required_future = [
        "future_rainfall_7d",
        "future_rainfall_14d",
        "future_rainfall_30d",
        "future_rainy_days_7d",
        "future_dry_days_7d",
    ]

    before = len(df)

    df = df.dropna(
        subset=required_future
    ).reset_index(drop=True)

    removed = before - len(df)

    print(
        f"\nRows removed because of incomplete "
        f"future windows: {removed:,}"
    )

    # -----------------------------------------------------
    # TARGET DISTRIBUTION
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("TARGET DISTRIBUTION")
    print("=" * 70)

    print(
        "\nONSET TARGET (7 DAYS)"
    )

    print(
        df["onset_target_7d"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nBREAK TARGET (7 DAYS)"
    )

    print(
        df["break_target_7d"]
        .value_counts()
        .sort_index()
    )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -----------------------------------------------------
    # FINAL REPORT
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("TARGET ENGINEERING COMPLETE")
    print("=" * 70)

    print(
        f"Rows       : {len(df):,}"
    )

    print(
        f"Columns    : {len(df.columns)}"
    )

    print(
        f"Grid cells : "
        f"{df[['latitude', 'longitude']].drop_duplicates().shape[0]}"
    )

    print(
        f"Date range : "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()