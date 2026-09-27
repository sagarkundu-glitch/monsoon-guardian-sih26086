import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path("data/climate/mjo/rmm.74toRealtime.txt")
OUTPUT_FILE = Path("data/climate/mjo/mjo_rmm_clean.csv")


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"MJO input file not found: {INPUT_FILE}"
    )


print("=" * 60)
print("MJO RMM DATA PROCESSING")
print("=" * 60)

print(f"Input file: {INPUT_FILE}")


# ============================================================
# READ BOM RMM FILE
# ============================================================

# The BOM file contains:
# year month day RMM1 RMM2 phase amplitude
# followed by a method description.

df = pd.read_csv(
    INPUT_FILE,
    skiprows=2,
    sep=r"\s+",
    header=None,
    usecols=[0, 1, 2, 3, 4, 5, 6],
    names=[
        "year",
        "month",
        "day",
        "rmm1",
        "rmm2",
        "phase",
        "amplitude",
    ],
    engine="python",
)


# ============================================================
# CONVERT DATA TYPES
# ============================================================

numeric_columns = [
    "year",
    "month",
    "day",
    "rmm1",
    "rmm2",
    "phase",
    "amplitude",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

df = df.dropna(
    subset=[
        "year",
        "month",
        "day",
        "rmm1",
        "rmm2",
    ]
).copy()


# ============================================================
# HANDLE BOM MISSING VALUES
# ============================================================

# BOM documentation indicates missing values can be:
# 1.E36 or 999.

for col in ["rmm1", "rmm2", "phase", "amplitude"]:
    df.loc[df[col] >= 900, col] = pd.NA


# ============================================================
# CREATE DATE
# ============================================================

df["date"] = pd.to_datetime(
    {
        "year": df["year"].astype(int),
        "month": df["month"].astype(int),
        "day": df["day"].astype(int),
    },
    errors="coerce",
)


# Remove rows with invalid dates

df = df.dropna(subset=["date"]).copy()


# ============================================================
# SELECT FINAL COLUMNS
# ============================================================

df = df[
    [
        "date",
        "rmm1",
        "rmm2",
        "phase",
        "amplitude",
    ]
].copy()


# ============================================================
# SORT AND REMOVE DUPLICATES
# ============================================================

df = df.sort_values("date")

df = df.drop_duplicates(
    subset=["date"],
    keep="last"
)


# ============================================================
# RESET INDEX
# ============================================================

df = df.reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("MJO PROCESSING COMPLETE")
print("-" * 60)

print(f"Rows: {len(df):,}")
print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
print(f"Missing RMM1: {df['rmm1'].isna().sum():,}")
print(f"Missing RMM2: {df['rmm2'].isna().sum():,}")
print(f"Missing Phase: {df['phase'].isna().sum():,}")
print(f"Missing Amplitude: {df['amplitude'].isna().sum():,}")

print()
print("2024 MJO DATA")
print("-" * 60)

mjo_2024 = df[
    (df["date"] >= "2024-01-01") &
    (df["date"] <= "2024-12-31")
].copy()

print(f"2024 rows: {len(mjo_2024):,}")

if len(mjo_2024) > 0:

    print()
    print("2024 RMM statistics:")
    print(
        mjo_2024[
            ["rmm1", "rmm2", "phase", "amplitude"]
        ].describe()
    )

print()
print(f"Saved to: {OUTPUT_FILE}")

print("=" * 60)