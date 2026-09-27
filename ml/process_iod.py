import pandas as pd
import os


INPUT_FILE = "data/climate/iod/dmi.had.long.csv"
OUTPUT_FILE = "data/climate/iod/iod_dmi_clean.csv"


def main():

    print("=" * 70)
    print("PROCESSING NOAA DMI / IOD DATA")
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
    # ---------------------------------------------------------
    date_column = df.columns[0]
    dmi_column = df.columns[1]

    df = df[[date_column, dmi_column]].copy()

    # Rename columns
    df.columns = [
        "date",
        "dmi"
    ]

    # ---------------------------------------------------------
    # Convert data types
    # ---------------------------------------------------------
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["dmi"] = pd.to_numeric(
        df["dmi"],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # NOAA missing value = -9999
    # ---------------------------------------------------------
    df.loc[
        df["dmi"] <= -9000,
        "dmi"
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
    # Add year and month
    # ---------------------------------------------------------
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # ---------------------------------------------------------
    # Extract 2024
    # ---------------------------------------------------------
    df_2024 = df[
        df["year"] == 2024
    ].copy()

    # ---------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------
    os.makedirs(
        "data/climate/iod",
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Save complete cleaned dataset
    # ---------------------------------------------------------
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("IOD / DMI DATA SUMMARY")
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
        "Missing DMI values:",
        df["dmi"].isna().sum()
    )

    print("\n2024 DMI DATA:")
    print("-" * 70)

    print(
        df_2024[
            ["date", "dmi"]
        ].to_string(index=False)
    )

    print(
        "\n2024 rows:",
        len(df_2024)
    )

    print(
        "2024 missing DMI:",
        df_2024["dmi"].isna().sum()
    )

    print(
        "2024 DMI mean:",
        round(
            df_2024["dmi"].mean(),
            4
        )
    )

    print(
        "2024 DMI minimum:",
        round(
            df_2024["dmi"].min(),
            4
        )
    )

    print(
        "2024 DMI maximum:",
        round(
            df_2024["dmi"].max(),
            4
        )
    )

    print("\nSaved cleaned dataset:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 70)
    print("IOD PROCESSING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()