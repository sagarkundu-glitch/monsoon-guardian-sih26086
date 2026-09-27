
from pathlib import Path
import pandas as pd
import xarray as xr

RAW_FILE = Path("data/raw/RF25_ind2024_rfp25.nc")
OUTPUT_FILE = Path("data/processed/imd_rainfall_west_bengal_2024.csv")

# Development pilot only; this is not yet block/Panchayat resolution.
MIN_LAT, MAX_LAT = 21.5, 27.2
MIN_LON, MAX_LON = 85.8, 89.9

def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"IMD file not found: {RAW_FILE}\n"
            "Put RF25_ind2024_rfp25.nc inside data/raw/."
        )

    print("Opening IMD NetCDF...")
    ds = xr.open_dataset(RAW_FILE)
    rain = ds["RAINFALL"]

    subset = rain.sel(
        LATITUDE=slice(MIN_LAT, MAX_LAT),
        LONGITUDE=slice(MIN_LON, MAX_LON),
    )

    print("Selected pilot region.")
    df = subset.to_dataframe(name="rainfall_mm").reset_index()

    df = df.rename(columns={
        "TIME": "date",
        "LATITUDE": "latitude",
        "LONGITUDE": "longitude",
    })

    df["date"] = pd.to_datetime(df["date"])
    df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce")

    # Negative values are treated as missing because rainfall is non-negative.
    df.loc[df["rainfall_mm"] < 0, "rainfall_mm"] = pd.NA

    df = df[
        ["date", "latitude", "longitude", "rainfall_mm"]
    ].sort_values(["date", "latitude", "longitude"])

    df = df.drop_duplicates(
        subset=["date", "latitude", "longitude"]
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    valid = df["rainfall_mm"].dropna()
    cells = df[["latitude", "longitude"]].drop_duplicates().shape[0]

    print("\n========================================")
    print("IMD INGESTION COMPLETE")
    print("========================================")
    print(f"Rows             : {len(df):,}")
    print(f"Grid cells       : {cells:,}")
    print(f"Date range       : {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Missing rainfall : {df['rainfall_mm'].isna().mean():.2%}")
    print(f"Min rainfall     : {valid.min():.2f} mm")
    print(f"Max rainfall     : {valid.max():.2f} mm")
    print(f"Mean rainfall    : {valid.mean():.2f} mm")
    print(f"Saved to         : {OUTPUT_FILE}")

    ds.close()

if __name__ == "__main__":
    main()
