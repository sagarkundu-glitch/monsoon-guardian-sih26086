import pandas as pd
import numpy as np


INPUT_FILE = "data/processed/imd_rainfall_features_2024.csv"
OUTPUT_FILE = "data/processed/imd_rainfall_multihorizon_2024.csv"

RAIN_THRESHOLD = 2.5


def main():

    print("=" * 70)
    print("CREATING MULTI-HORIZON TARGETS")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load feature dataset
    # ---------------------------------------------------------
    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["latitude", "longitude", "date"]
    ).reset_index(drop=True)

    print(f"\nInput rows: {len(df):,}")
    print(
        "Grid cells:",
        df[["latitude", "longitude"]].drop_duplicates().shape[0]
    )

    # ---------------------------------------------------------
    # Create daily rainy/dry indicators
    # ---------------------------------------------------------
    df["is_rainy"] = (
        df["rainfall_1d"] >= RAIN_THRESHOLD
    ).astype(int)

    df["is_dry"] = (
        df["rainfall_1d"] < RAIN_THRESHOLD
    ).astype(int)

    # ---------------------------------------------------------
    # Horizons
    # ---------------------------------------------------------
    horizons = [7, 14, 30]

    # ---------------------------------------------------------
    # Process each grid cell separately
    # This avoids fragile MultiIndex operations.
    # ---------------------------------------------------------
    result_parts = []

    for (lat, lon), group in df.groupby(
        ["latitude", "longitude"],
        sort=False
    ):

        group = group.sort_values("date").copy()

        rainy = group["is_rainy"].to_numpy()
        dry = group["is_dry"].to_numpy()
        rainfall = group["rainfall_1d"].to_numpy()

        n = len(group)

        for h in horizons:

            future_rainy = np.full(n, np.nan)
            future_dry = np.full(n, np.nan)
            future_rainfall = np.full(n, np.nan)

            # Future window starts TOMORROW.
            # Current day's rainfall is NOT included.
            for i in range(n):

                start = i + 1
                end = i + 1 + h

                if end <= n:

                    future_rainy[i] = np.sum(
                        rainy[start:end]
                    )

                    future_dry[i] = np.sum(
                        dry[start:end]
                    )

                    future_rainfall[i] = np.sum(
                        rainfall[start:end]
                    )

            # -------------------------------------------------
            # Store future statistics
            # -------------------------------------------------
            group[f"future_{h}d_rainy_days"] = future_rainy
            group[f"future_{h}d_dry_days"] = future_dry
            group[f"future_{h}d_rainfall_mm"] = future_rainfall

            # -------------------------------------------------
            # ONSET TARGET
            #
            # 7 days  -> at least 4 rainy days
            # 14 days -> at least 8 rainy days
            # 30 days -> at least 15 rainy days
            # -------------------------------------------------
            onset_threshold = {
                7: 4,
                14: 8,
                30: 15
            }[h]

            group[f"onset_target_{h}d"] = np.where(
                np.isnan(future_rainy),
                np.nan,
                (
                    future_rainy >= onset_threshold
                ).astype(int)
            )

            # -------------------------------------------------
            # BREAK TARGET
            #
            # 7 days  -> at least 5 dry days
            # 14 days -> at least 10 dry days
            # 30 days -> at least 21 dry days
            # -------------------------------------------------
            break_threshold = {
                7: 5,
                14: 10,
                30: 21
            }[h]

            group[f"break_target_{h}d"] = np.where(
                np.isnan(future_dry),
                np.nan,
                (
                    future_dry >= break_threshold
                ).astype(int)
            )

        result_parts.append(group)

    # ---------------------------------------------------------
    # Combine all grid cells
    # ---------------------------------------------------------
    result = pd.concat(
        result_parts,
        ignore_index=True
    )

    # ---------------------------------------------------------
    # Remove helper columns
    # ---------------------------------------------------------
    result = result.drop(
        columns=["is_rainy", "is_dry"]
    )

    # ---------------------------------------------------------
    # Remove rows without complete 30-day future window
    # ---------------------------------------------------------
    target_columns = [
        "onset_target_7d",
        "onset_target_14d",
        "onset_target_30d",
        "break_target_7d",
        "break_target_14d",
        "break_target_30d",
    ]

    before = len(result)

    result = result.dropna(
        subset=target_columns
    ).copy()

    removed = before - len(result)

    # ---------------------------------------------------------
    # Convert target columns to integers
    # ---------------------------------------------------------
    for col in target_columns:
        result[col] = result[col].astype(int)

    # ---------------------------------------------------------
    # Sort final dataset
    # ---------------------------------------------------------
    result = result.sort_values(
        ["date", "latitude", "longitude"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("TARGET SUMMARY")
    print("=" * 70)

    print(f"\nRows removed for incomplete future windows: {removed:,}")

    print(
        "\nFinal rows:",
        f"{len(result):,}"
    )

    print(
        "Grid cells:",
        result[["latitude", "longitude"]]
        .drop_duplicates()
        .shape[0]
    )

    print(
        "Date range:",
        result["date"].min().date(),
        "→",
        result["date"].max().date()
    )

    # ---------------------------------------------------------
    # Target distributions
    # ---------------------------------------------------------
    print("\nONSET TARGETS")
    print("-" * 70)

    for h in horizons:

        col = f"onset_target_{h}d"

        counts = result[col].value_counts().sort_index()

        print(f"\n{h}-day onset:")
        print(counts)

    print("\nBREAK TARGETS")
    print("-" * 70)

    for h in horizons:

        col = f"break_target_{h}d"

        counts = result[col].value_counts().sort_index()

        print(f"\n{h}-day break:")
        print(counts)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("MULTI-HORIZON TARGET CREATION COMPLETE")
    print("=" * 70)

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()