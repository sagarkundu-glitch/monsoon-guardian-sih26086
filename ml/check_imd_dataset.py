
from pathlib import Path
import pandas as pd

FILE = Path("data/processed/imd_rainfall_west_bengal_2024.csv")

if not FILE.exists():
    raise FileNotFoundError(
        "Processed file not found. Run: python ml/ingest_imd_rainfall.py"
    )

df = pd.read_csv(FILE, parse_dates=["date"])

print("========================================")
print("IMD DATA QUALITY CHECK")
print("========================================")
print("Rows:", f"{len(df):,}")
print("Columns:", list(df.columns))
print("Dates:", df["date"].min().date(), "to", df["date"].max().date())
print("Grid cells:", df[["latitude", "longitude"]].drop_duplicates().shape[0])
print("Missing rainfall:", f"{df['rainfall_mm'].isna().mean():.2%}")
print("\nRainfall statistics:")
print(df["rainfall_mm"].describe())
print("\nSample:")
print(df.head(10).to_string(index=False))
