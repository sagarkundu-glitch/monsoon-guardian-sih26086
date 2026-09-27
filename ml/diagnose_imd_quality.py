from pathlib import Path
import pandas as pd

FILE = Path("data/processed/imd_rainfall_west_bengal_2024.csv")

if not FILE.exists():
    raise FileNotFoundError(
        "Processed dataset not found. "
        "Run the IMD ingestion script first."
    )

df = pd.read_csv(FILE, parse_dates=["date"])

print("=" * 55)
print("SIH26086 — IMD DATA QUALITY DIAGNOSTICS")
print("=" * 55)

# ---------------------------------------------------------
# 1. BASIC INFORMATION
# ---------------------------------------------------------

print("\n[1] BASIC INFORMATION")

print("Rows:", f"{len(df):,}")
print("Columns:", list(df.columns))

print(
    "Date range:",
    df["date"].min().date(),
    "to",
    df["date"].max().date()
)

print(
    "Unique dates:",
    df["date"].nunique()
)

print(
    "Grid cells:",
    df[["latitude", "longitude"]]
    .drop_duplicates()
    .shape[0]
)

# ---------------------------------------------------------
# 2. MISSING VALUES
# ---------------------------------------------------------

print("\n[2] MISSING VALUES")

missing = df["rainfall_mm"].isna().sum()
total = len(df)

print("Missing rainfall:", f"{missing:,}")
print("Missing percentage:", f"{missing / total:.2%}")

# ---------------------------------------------------------
# 3. VALID VALUES
# ---------------------------------------------------------

print("\n[3] VALID RAINFALL")

valid = df["rainfall_mm"].dropna()

print("Valid observations:", f"{len(valid):,}")
print("Minimum:", f"{valid.min():.2f} mm")
print("Maximum:", f"{valid.max():.2f} mm")
print("Mean:", f"{valid.mean():.2f} mm")
print("Median:", f"{valid.median():.2f} mm")

# ---------------------------------------------------------
# 4. NEGATIVE VALUES
# ---------------------------------------------------------

print("\n[4] NEGATIVE VALUES")

negative = (df["rainfall_mm"] < 0).sum()

print("Negative rainfall values:", negative)

# ---------------------------------------------------------
# 5. DUPLICATES
# ---------------------------------------------------------

print("\n[5] DUPLICATES")

duplicates = df.duplicated(
    subset=["date", "latitude", "longitude"]
).sum()

print("Duplicate observations:", duplicates)

# ---------------------------------------------------------
# 6. MISSING BY GRID CELL
# ---------------------------------------------------------

print("\n[6] GRID CELLS WITH MOST MISSING DATA")

cell_report = (
    df.groupby(["latitude", "longitude"])
    .agg(
        total_days=("rainfall_mm", "size"),
        missing_days=("rainfall_mm", lambda x: x.isna().sum())
    )
)

cell_report["missing_percentage"] = (
    cell_report["missing_days"] /
    cell_report["total_days"]
)

print(
    cell_report
    .sort_values("missing_percentage", ascending=False)
    .head(15)
    .to_string()
)

# ---------------------------------------------------------
# 7. MISSING BY DATE
# ---------------------------------------------------------

print("\n[7] DATES WITH MOST MISSING DATA")

date_report = (
    df.groupby("date")
    .agg(
        total_cells=("rainfall_mm", "size"),
        missing_cells=("rainfall_mm", lambda x: x.isna().sum())
    )
)

date_report["missing_percentage"] = (
    date_report["missing_cells"] /
    date_report["total_cells"]
)

print(
    date_report
    .sort_values("missing_percentage", ascending=False)
    .head(15)
    .to_string()
)

# ---------------------------------------------------------
# 8. RAINY DAYS
# ---------------------------------------------------------

print("\n[8] RAINFALL OCCURRENCE")

valid = df["rainfall_mm"].dropna()

print(
    "Zero-rainfall observations:",
    f"{(valid == 0).sum():,}"
)

print(
    "Positive-rainfall observations:",
    f"{(valid > 0).sum():,}"
)

print(
    "Positive rainfall percentage:",
    f"{(valid > 0).mean():.2%}"
)

# ---------------------------------------------------------
# FINAL
# ---------------------------------------------------------

print("\n" + "=" * 55)
print("DIAGNOSTIC COMPLETE")
print("=" * 55)