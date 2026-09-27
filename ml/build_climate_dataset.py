import pandas as pd
from pathlib import Path


# ============================================================
# FILE PATHS
# ============================================================

RAINFALL_FILE = Path(
    "data/processed/imd_rainfall_training_2024.csv"
)

ENSO_FILE = Path(
    "data/climate/enso/enso_meiv2_clean.csv"
)

IOD_FILE = Path(
    "data/climate/iod/iod_dmi_clean.csv"
)

MJO_FILE = Path(
    "data/climate/mjo/mjo_rmm_clean.csv"
)

OUTPUT_FILE = Path(
    "data/processed/imd_rainfall_climate_training_2024.csv"
)


print("=" * 70)
print("BUILDING CLIMATE-AWARE TRAINING DATASET")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

files = {
    "Rainfall": RAINFALL_FILE,
    "ENSO": ENSO_FILE,
    "IOD": IOD_FILE,
    "MJO": MJO_FILE,
}

for name, path in files.items():
    if not path.exists():
        raise FileNotFoundError(
            f"{name} file not found: {path}"
        )

    print(f"✓ {name}: {path}")


# ============================================================
# LOAD DATA
# ============================================================

rainfall = pd.read_csv(RAINFALL_FILE)
enso = pd.read_csv(ENSO_FILE)
iod = pd.read_csv(IOD_FILE)
mjo = pd.read_csv(MJO_FILE)


print()
print("Files loaded successfully.")


# ============================================================
# CONVERT DATES
# ============================================================

rainfall["date"] = pd.to_datetime(
    rainfall["date"],
    errors="coerce"
)

enso["date"] = pd.to_datetime(
    enso["date"],
    errors="coerce"
)

iod["date"] = pd.to_datetime(
    iod["date"],
    errors="coerce"
)

mjo["date"] = pd.to_datetime(
    mjo["date"],
    errors="coerce"
)


# ============================================================
# KEEP ONLY 2024 CLIMATE DATA
# ============================================================

enso_2024 = enso[
    (enso["date"] >= "2024-01-01") &
    (enso["date"] <= "2024-12-31")
].copy()

iod_2024 = iod[
    (iod["date"] >= "2024-01-01") &
    (iod["date"] <= "2024-12-31")
].copy()

mjo_2024 = mjo[
    (mjo["date"] >= "2024-01-01") &
    (mjo["date"] <= "2024-12-31")
].copy()


print()
print("2024 climate records:")
print(f"ENSO: {len(enso_2024)}")
print(f"IOD : {len(iod_2024)}")
print(f"MJO : {len(mjo_2024)}")


# ============================================================
# PREPARE YEAR + MONTH
# ============================================================

rainfall["year"] = rainfall["date"].dt.year
rainfall["month"] = rainfall["date"].dt.month

enso_2024["year"] = enso_2024["date"].dt.year
enso_2024["month"] = enso_2024["date"].dt.month

iod_2024["year"] = iod_2024["date"].dt.year
iod_2024["month"] = iod_2024["date"].dt.month


# ============================================================
# PREPARE ENSO
# ============================================================

# Keep only the required columns.
enso_2024 = enso_2024[
    [
        "year",
        "month",
        "mei_v2",
    ]
].copy()


# ============================================================
# PREPARE IOD
# ============================================================

iod_2024 = iod_2024[
    [
        "year",
        "month",
        "dmi",
    ]
].copy()


# ============================================================
# MERGE MONTHLY ENSO
# ============================================================

rainfall = rainfall.merge(
    enso_2024,
    on=["year", "month"],
    how="left",
    validate="many_to_one"
)


print()
print("✓ ENSO merged")


# ============================================================
# MERGE MONTHLY IOD
# ============================================================

rainfall = rainfall.merge(
    iod_2024,
    on=["year", "month"],
    how="left",
    validate="many_to_one"
)


print("✓ IOD merged")


# ============================================================
# PREPARE MJO
# ============================================================

mjo_2024 = mjo_2024[
    [
        "date",
        "rmm1",
        "rmm2",
        "phase",
        "amplitude",
    ]
].copy()


# Make sure there is only one MJO record per date.

mjo_2024 = mjo_2024.drop_duplicates(
    subset=["date"],
    keep="last"
)


# ============================================================
# MERGE DAILY MJO
# ============================================================

rainfall = rainfall.merge(
    mjo_2024,
    on="date",
    how="left",
    validate="many_to_one"
)


print("✓ MJO merged")


# ============================================================
# REMOVE TEMPORARY COLUMNS
# ============================================================

rainfall = rainfall.drop(
    columns=["year", "month"]
)


# ============================================================
# SORT DATA
# ============================================================

sort_columns = [
    "date",
    "latitude",
    "longitude",
]

rainfall = rainfall.sort_values(
    sort_columns
).reset_index(drop=True)


# ============================================================
# CHECK DUPLICATES
# ============================================================

duplicate_count = rainfall.duplicated(
    subset=["date", "latitude", "longitude"]
).sum()


print()
print("Duplicate grid-date records:", duplicate_count)


if duplicate_count > 0:
    raise ValueError(
        "Duplicate date/grid records detected."
    )


# ============================================================
# CHECK MISSING CLIMATE VALUES
# ============================================================

climate_columns = [
    "mei_v2",
    "dmi",
    "rmm1",
    "rmm2",
    "phase",
    "amplitude",
]


print()
print("CLIMATE DATA QUALITY")
print("-" * 70)

for col in climate_columns:

    missing = rainfall[col].isna().sum()

    print(
        f"{col:12s} missing: "
        f"{missing:,} "
        f"({missing / len(rainfall) * 100:.2f}%)"
    )


# ============================================================
# CHECK 2024 CLIMATE VALUES
# ============================================================

if rainfall[climate_columns].isna().any().any():

    print()
    print("WARNING: Missing climate values detected.")

    # Since climate data should cover the full 2024 period,
    # stop here rather than silently filling values.

    raise ValueError(
        "Climate merge produced missing values. "
        "Check the climate datasets before training."
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

rainfall.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("CLIMATE DATASET CREATED SUCCESSFULLY")
print("=" * 70)

print(f"Rows       : {len(rainfall):,}")
print(f"Columns    : {len(rainfall.columns)}")
print(
    f"Grid cells : "
    f"{rainfall[['latitude', 'longitude']].drop_duplicates().shape[0]:,}"
)

print(
    f"Date range : "
    f"{rainfall['date'].min().date()} "
    f"-> "
    f"{rainfall['date'].max().date()}"
)

print()
print("Climate features:")
for col in climate_columns:
    print(f"  ✓ {col}")

print()
print(f"Saved to:")
print(OUTPUT_FILE)

print("=" * 70)