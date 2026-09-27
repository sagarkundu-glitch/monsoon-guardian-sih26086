from pathlib import Path
import pandas as pd

FILE = Path("data/processed/imd_rainfall_west_bengal_2024.csv")

df = pd.read_csv(FILE, parse_dates=["date"])

print("=" * 60)
print("SIH26086 — MISSING GRID CELL ANALYSIS")
print("=" * 60)

# Find grid cells that contain missing rainfall at any time.
missing_cells = (
    df[df["rainfall_mm"].isna()]
    [["latitude", "longitude"]]
    .drop_duplicates()
    .sort_values(["latitude", "longitude"])
)

print("\nTotal missing grid cells:", len(missing_cells))

print("\nMissing grid-cell coordinates:")
print(missing_cells.to_string(index=False))

# ---------------------------------------------------------
# Check whether these cells are missing for the whole year
# ---------------------------------------------------------

cell_stats = (
    df.groupby(["latitude", "longitude"])
    .agg(
        total_days=("rainfall_mm", "size"),
        missing_days=("rainfall_mm", lambda x: x.isna().sum()),
        valid_days=("rainfall_mm", lambda x: x.notna().sum())
    )
    .reset_index()
)

cell_stats["missing_percentage"] = (
    cell_stats["missing_days"] /
    cell_stats["total_days"] *
    100
)

print("\n" + "=" * 60)
print("CELLS WITH 100% MISSING DATA")
print("=" * 60)

always_missing = cell_stats[
    cell_stats["missing_days"] == cell_stats["total_days"]
]

print("Count:", len(always_missing))

print(
    always_missing[
        ["latitude", "longitude", "total_days", "missing_days"]
    ].to_string(index=False)
)

print("\n" + "=" * 60)
print("CELLS WITH PARTIAL MISSING DATA")
print("=" * 60)

partial = cell_stats[
    (cell_stats["missing_days"] > 0) &
    (cell_stats["missing_days"] < cell_stats["total_days"])
]

print("Count:", len(partial))

if len(partial) > 0:
    print(partial.to_string(index=False))
else:
    print("No partially missing cells found.")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)