import pandas as pd
from pathlib import Path


# ============================================================
# FILE PATHS
# ============================================================

RAINFALL_FILE = Path(
    "data/processed/imd_rainfall_multihorizon_2024.csv"
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
    "data/processed/imd_rainfall_climate_multihorizon_2024.csv"
)


print("=" * 70)
print("BUILDING CLIMATE MULTIHORIZON DATASET")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

for name, path in {
    "Rainfall multihorizon": RAINFALL_FILE,
    "ENSO": ENSO_FILE,
    "IOD": IOD_FILE,
    "MJO": MJO_FILE,
}.items():

    if not path.exists():
        raise FileNotFoundError(
            f"{name} file not found: {path}"
        )

    print(f"✓ {name}: {path}")


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(RAINFALL_FILE)
enso = pd.read_csv(ENSO_FILE)
iod = pd.read_csv(IOD_FILE)
mjo = pd.read_csv(MJO_FILE)


print()
print(f"Rainfall rows: {len(df):,}")
print(f"Rainfall columns: {len(df.columns)}")


# ============================================================
# DATE CONVERSION
# ============================================================

df["date"] = pd.to_datetime(
    df["date"],
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
# VERIFY SIX TARGETS
# ============================================================

targets = [
    "onset_target_7d",
    "onset_target_14d",
    "onset_target_30d",
    "break_target_7d",
    "break_target_14d",
    "break_target_30d",
]

missing_targets = [
    target
    for target in targets
    if target not in df.columns
]

if missing_targets:
    raise ValueError(
        "The multihorizon rainfall dataset is missing:\n"
        + "\n".join(missing_targets)
    )

print()
print("✓ All six prediction targets found")


# ============================================================
# YEAR + MONTH
# ============================================================

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

enso["year"] = enso["date"].dt.year
enso["month"] = enso["date"].dt.month

iod["year"] = iod["date"].dt.year
iod["month"] = iod["date"].dt.month


# ============================================================
# ENSO
# ============================================================

enso = enso[
    (enso["date"] >= "2024-01-01") &
    (enso["date"] <= "2024-12-31")
].copy()

enso = enso[
    [
        "year",
        "month",
        "mei_v2",
    ]
].drop_duplicates(
    subset=["year", "month"]
)


# ============================================================
# IOD
# ============================================================

iod = iod[
    (iod["date"] >= "2024-01-01") &
    (iod["date"] <= "2024-12-31")
].copy()

iod = iod[
    [
        "year",
        "month",
        "dmi",
    ]
].drop_duplicates(
    subset=["year", "month"]
)


# ============================================================
# MERGE ENSO
# ============================================================

df = df.merge(
    enso,
    on=["year", "month"],
    how="left",
    validate="many_to_one"
)

print("✓ ENSO merged")


# ============================================================
# MERGE IOD
# ============================================================

df = df.merge(
    iod,
    on=["year", "month"],
    how="left",
    validate="many_to_one"
)

print("✓ IOD merged")


# ============================================================
# MJO
# ============================================================

mjo = mjo[
    (mjo["date"] >= "2024-01-01") &
    (mjo["date"] <= "2024-12-31")
].copy()

mjo = mjo[
    [
        "date",
        "rmm1",
        "rmm2",
        "phase",
        "amplitude",
    ]
].drop_duplicates(
    subset=["date"]
)


# ============================================================
# MERGE MJO
# ============================================================

df = df.merge(
    mjo,
    on="date",
    how="left",
    validate="many_to_one"
)

print("✓ MJO merged")


# ============================================================
# REMOVE TEMPORARY COLUMNS
# ============================================================

df = df.drop(
    columns=["year", "month"]
)


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    [
        "date",
        "latitude",
        "longitude",
    ]
).reset_index(drop=True)


# ============================================================
# CHECK DUPLICATES
# ============================================================

duplicates = df.duplicated(
    subset=[
        "date",
        "latitude",
        "longitude",
    ]
).sum()

print()
print(
    f"Duplicate date/grid records: {duplicates}"
)

if duplicates > 0:
    raise ValueError(
        "Duplicate date/grid records detected."
    )


# ============================================================
# CLIMATE FEATURES
# ============================================================

climate_features = [
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

for column in climate_features:

    missing = df[column].isna().sum()

    print(
        f"{column:12s}: "
        f"{missing:,} missing "
        f"({missing / len(df) * 100:.2f}%)"
    )


# ============================================================
# STOP IF MISSING
# ============================================================

if df[climate_features].isna().any().any():

    raise ValueError(
        "Missing climate values detected. "
        "Do not continue to model training."
    )


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
print("=" * 70)
print("CLIMATE MULTIHORIZON DATASET CREATED")
print("=" * 70)

print(f"Rows       : {len(df):,}")
print(f"Columns    : {len(df.columns)}")

print(
    "Grid cells : "
    f"{df[['latitude', 'longitude']].drop_duplicates().shape[0]:,}"
)

print(
    "Date range : "
    f"{df['date'].min().date()} -> "
    f"{df['date'].max().date()}"
)

print()
print("Targets:")

for target in targets:
    print(f"  ✓ {target}")

print()
print("Climate features:")

for feature in climate_features:
    print(f"  ✓ {feature}")

print()
print("Saved to:")
print(OUTPUT_FILE)

print("=" * 70)