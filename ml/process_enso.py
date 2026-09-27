import pandas as pd
import os


INPUT_FILE = "data/climate/enso/meiv2.csv"
OUTPUT_FILE = "data/climate/enso/enso_meiv2_clean.csv"


def main():

    print("=" * 70)
    print("PROCESSING NOAA MEI / ENSO DATA")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load NOAA CSV
    # ---------------------------------------------------------
    df = pd.read_csv(INPUT_FILE)

    print("\nOriginal columns:")
    print(df.columns.tolist())

    print("\nOriginal rows:", len(df))

    # ---------------------------------------------------------
    # Use first two columns
    # NOAA file contains:
    # Date + MEI.v2 value
    # ---------------------------------------------------------
    date_column = df.columns[0]
    mei_column = df.columns[1]

    df = df[[date_column, mei_column]].copy()

    # Rename to simple names
    df.columns = [
        "date",
        "mei_v2"
    ]

    # ---------------------------------------------------------
    # Convert data types
    # ---------------------------------------------------------
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["mei_v2"] = pd.to_numeric(
        df["mei_v2"],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # NOAA missing value
    # ---------------------------------------------------------
    df.loc[
        df["mei_v2"] <= -900,
        "mei_v2"
    ] = pd.NA

    # ---------------------------------------------------------
    # Remove invalid dates
    # ---------------------------------------------------------
    df = df.dropna(
        subset=["date"]
    ).copy()

    # ---------------------------------------------------------
    # Sort by date
    # ---------------------------------------------------------
    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Keep useful columns
    # ---------------------------------------------------------
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # ---------------------------------------------------------
    # 2024 subset
    # ---------------------------------------------------------
    df_2024 = df[
        df["year"] == 2024
    ].copy()

    # ---------------------------------------------------------
    # Create directory if needed
    # ---------------------------------------------------------
    os.makedirs(
        "data/climate/enso",
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Save complete cleaned dataset
    # ---------------------------------------------------------
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("ENSO DATA SUMMARY")
    print("=" * 70)

    print(
        "\nFull date range:",
        df["date"].min().date(),
        "→",
        df["date"].max().date()
    )

    print(
        "Full rows:",
        len(df)
    )

    print(
        "Missing MEI values:",
        df["mei_v2"].isna().sum()
    )

    print(
        "\n2024 MEI DATA:"
    )

    print(
        df_2024[
            ["date", "mei_v2"]
        ].to_string(index=False)
    )

    print("\n2024 rows:", len(df_2024))

    print(
        "\n2024 missing MEI:",
        df_2024["mei_v2"].isna().sum()
    )

    print(
        "\n2024 MEI mean:",
        round(
            df_2024["mei_v2"].mean(),
            4
        )
    )

    print(
        "2024 MEI minimum:",
        round(
            df_2024["mei_v2"].min(),
            4
        )
    )

    print(
        "2024 MEI maximum:",
        round(
            df_2024["mei_v2"].max(),
            4
        )
    )

    print("\nSaved cleaned dataset:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 70)
    print("ENSO PROCESSING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()