from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/imd_rainfall_clean_2024.csv"
)

OUTPUT_FILE = Path(
    "data/processed/imd_rainfall_features_2024.csv"
)

RAINY_DAY_THRESHOLD = 0.1
DRY_DAY_THRESHOLD = 0.1


def main():

    print("=" * 70)
    print("SIH26086 — RAINFALL FEATURE ENGINEERING")
    print("=" * 70)

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    print("\nLoading clean IMD dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"]
    )

    print(f"Input rows: {len(df):,}")

    # ---------------------------------------------------------
    # SORT
    # ---------------------------------------------------------

    df = df.sort_values(
        ["latitude", "longitude", "date"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # CREATE GRID-CELL GROUP
    # ---------------------------------------------------------

    group_columns = ["latitude", "longitude"]

    grouped = df.groupby(
        group_columns,
        sort=False
    )

    rainfall = df["rainfall_mm"]

    # ---------------------------------------------------------
    # 1-DAY RAINFALL
    # ---------------------------------------------------------

    df["rainfall_1d"] = rainfall

    # ---------------------------------------------------------
    # ROLLING RAINFALL
    # ---------------------------------------------------------

    print("\nCalculating rolling rainfall totals...")

    df["rainfall_3d"] = (
        grouped["rainfall_mm"]
        .transform(
            lambda x: x.rolling(
                window=3,
                min_periods=3
            ).sum()
        )
    )

    df["rainfall_7d"] = (
        grouped["rainfall_mm"]
        .transform(
            lambda x: x.rolling(
                window=7,
                min_periods=7
            ).sum()
        )
    )

    df["rainfall_14d"] = (
        grouped["rainfall_mm"]
        .transform(
            lambda x: x.rolling(
                window=14,
                min_periods=14
            ).sum()
        )
    )

    df["rainfall_30d"] = (
        grouped["rainfall_mm"]
        .transform(
            lambda x: x.rolling(
                window=30,
                min_periods=30
            ).sum()
        )
    )

    # ---------------------------------------------------------
    # RAINY DAY
    # ---------------------------------------------------------

    print("\nCalculating rainy-day features...")

    df["_rainy"] = (
        rainfall > RAINY_DAY_THRESHOLD
    ).astype(int)

    rainy_grouped = df.groupby(
        group_columns,
        sort=False
    )

    df["rainy_days_7d"] = (
        rainy_grouped["_rainy"]
        .transform(
            lambda x: x.rolling(
                window=7,
                min_periods=7
            ).sum()
        )
    )

    df["rainy_days_14d"] = (
        rainy_grouped["_rainy"]
        .transform(
            lambda x: x.rolling(
                window=14,
                min_periods=14
            ).sum()
        )
    )

    df["rainy_days_30d"] = (
        rainy_grouped["_rainy"]
        .transform(
            lambda x: x.rolling(
                window=30,
                min_periods=30
            ).sum()
        )
    )

    # ---------------------------------------------------------
    # DRY DAYS
    # ---------------------------------------------------------

    print("\nCalculating dry-day features...")

    df["_dry"] = (
        rainfall <= DRY_DAY_THRESHOLD
    ).astype(int)

    dry_grouped = df.groupby(
        group_columns,
        sort=False
    )

    df["dry_days_7d"] = (
        dry_grouped["_dry"]
        .transform(
            lambda x: x.rolling(
                window=7,
                min_periods=7
            ).sum()
        )
    )

    df["dry_days_14d"] = (
        dry_grouped["_dry"]
        .transform(
            lambda x: x.rolling(
                window=14,
                min_periods=14
            ).sum()
        )
    )

    df["dry_days_30d"] = (
        dry_grouped["_dry"]
        .transform(
            lambda x: x.rolling(
                window=30,
                min_periods=30
            ).sum()
        )
    )

    # ---------------------------------------------------------
    # CONSECUTIVE DRY DAYS
    # ---------------------------------------------------------

    print("\nCalculating consecutive dry days...")

    def dry_streak(series):

        result = []
        count = 0

        for value in series:

            if value <= DRY_DAY_THRESHOLD:
                count += 1
            else:
                count = 0

            result.append(count)

        return pd.Series(
            result,
            index=series.index
        )

    df["consecutive_dry_days"] = (
        grouped["rainfall_mm"]
        .transform(dry_streak)
    )

    # ---------------------------------------------------------
    # LAG FEATURES
    # ---------------------------------------------------------

    print("\nCalculating lag features...")

    df["rainfall_lag_1d"] = (
        grouped["rainfall_mm"]
        .shift(1)
    )

    df["rainfall_lag_3d"] = (
        grouped["rainfall_mm"]
        .shift(3)
    )

    df["rainfall_lag_7d"] = (
        grouped["rainfall_mm"]
        .shift(7)
    )

    # ---------------------------------------------------------
    # RAINFALL CHANGE
    # ---------------------------------------------------------

    df["rainfall_change_1d"] = (
        df["rainfall_mm"] -
        df["rainfall_lag_1d"]
    )

    # ---------------------------------------------------------
    # REMOVE TEMPORARY COLUMNS
    # ---------------------------------------------------------

    df = df.drop(
        columns=[
            "_rainy",
            "_dry"
        ]
    )

    # ---------------------------------------------------------
    # REQUIRED FEATURES
    # ---------------------------------------------------------

    required_features = [
        "rainfall_3d",
        "rainfall_7d",
        "rainfall_14d",
        "rainfall_30d",

        "rainy_days_7d",
        "rainy_days_14d",
        "rainy_days_30d",

        "dry_days_7d",
        "dry_days_14d",
        "dry_days_30d",

        "rainfall_lag_1d",
        "rainfall_lag_3d",
        "rainfall_lag_7d",
    ]

    # ---------------------------------------------------------
    # REMOVE INITIAL INCOMPLETE WINDOWS
    # ---------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=required_features
    ).reset_index(drop=True)

    removed = before - len(df)

    print(
        f"\nRows removed because of incomplete "
        f"initial windows: {removed:,}"
    )

    # ---------------------------------------------------------
    # FINAL COLUMN ORDER
    # ---------------------------------------------------------

    columns = [
        "date",
        "latitude",
        "longitude",

        "rainfall_mm",

        "rainfall_1d",
        "rainfall_3d",
        "rainfall_7d",
        "rainfall_14d",
        "rainfall_30d",

        "rainy_days_7d",
        "rainy_days_14d",
        "rainy_days_30d",

        "dry_days_7d",
        "dry_days_14d",
        "dry_days_30d",

        "consecutive_dry_days",

        "rainfall_lag_1d",
        "rainfall_lag_3d",
        "rainfall_lag_7d",

        "rainfall_change_1d",
    ]

    df = df[columns]

    # ---------------------------------------------------------
    # FINAL VALIDATION
    # ---------------------------------------------------------

    print("\nChecking final dataset...")

    if "latitude" not in df.columns:
        raise RuntimeError(
            "Latitude column missing."
        )

    if "longitude" not in df.columns:
        raise RuntimeError(
            "Longitude column missing."
        )

    if df["rainfall_mm"].isna().any():
        raise RuntimeError(
            "Unexpected missing rainfall values."
        )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # REPORT
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)

    print(
        f"Rows          : {len(df):,}"
    )

    print(
        f"Columns       : {len(df.columns)}"
    )

    print(
        f"Grid cells    : "
        f"{df[['latitude', 'longitude']].drop_duplicates().shape[0]}"
    )

    print(
        f"Date range    : "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    print(
        f"Missing values: "
        f"{df.isna().sum().sum():,}"
    )

    print("\nFeatures created:")

    for column in columns[4:]:
        print(f"  ✓ {column}")

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()