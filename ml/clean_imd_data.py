from pathlib import Path
import pandas as pd

INPUT_FILE = Path(
    "data/processed/imd_rainfall_west_bengal_2024.csv"
)

OUTPUT_FILE = Path(
    "data/processed/imd_rainfall_clean_2024.csv"
)


def main():

    print("=" * 60)
    print("SIH26086 — IMD RAINFALL CLEANING")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    print("\nLoading dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"]
    )

    print(f"Original rows: {len(df):,}")

    # --------------------------------------------------
    # Identify grid cells
    # --------------------------------------------------

    cell_stats = (
        df.groupby(
            ["latitude", "longitude"]
        )
        .agg(
            total_days=("rainfall_mm", "size"),
            missing_days=(
                "rainfall_mm",
                lambda x: x.isna().sum()
            )
        )
        .reset_index()
    )

    # Cells with rainfall data available for at least
    # one observation.
    usable_cells = cell_stats[
        cell_stats["missing_days"]
        < cell_stats["total_days"]
    ]

    print(
        f"Original grid cells: "
        f"{len(cell_stats)}"
    )

    print(
        f"Usable grid cells: "
        f"{len(usable_cells)}"
    )

    print(
        f"Permanent missing cells: "
        f"{len(cell_stats) - len(usable_cells)}"
    )

    # --------------------------------------------------
    # Keep only usable grid cells
    # --------------------------------------------------

    df = df.merge(
        usable_cells[
            ["latitude", "longitude"]
        ],
        on=["latitude", "longitude"],
        how="inner"
    )

    # --------------------------------------------------
    # Safety checks
    # --------------------------------------------------

    negative_values = (
        df["rainfall_mm"] < 0
    ).sum()

    if negative_values > 0:
        print(
            f"WARNING: {negative_values} "
            "negative values found."
        )

        df.loc[
            df["rainfall_mm"] < 0,
            "rainfall_mm"
        ] = pd.NA

    duplicates = df.duplicated(
        subset=[
            "date",
            "latitude",
            "longitude"
        ]
    ).sum()

    print(
        f"Duplicate observations: "
        f"{duplicates}"
    )

    if duplicates > 0:
        df = df.drop_duplicates(
            subset=[
                "date",
                "latitude",
                "longitude"
            ]
        )

    # --------------------------------------------------
    # Sort
    # --------------------------------------------------

    df = df.sort_values(
        [
            "latitude",
            "longitude",
            "date"
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------
    # Final report
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)

    print(
        f"Clean rows: "
        f"{len(df):,}"
    )

    print(
        f"Grid cells: "
        f"{df[['latitude', 'longitude']].drop_duplicates().shape[0]}"
    )

    print(
        f"Remaining missing rainfall: "
        f"{df['rainfall_mm'].isna().sum():,}"
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()