import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)


# ============================================================
# INPUT
# ============================================================

INPUT_FILE = Path(
    "data/processed/imd_rainfall_climate_multihorizon_2024.csv"
)

OUTPUT_FILE = Path(
    "data/processed/climate_ablation_results_2024.csv"
)


# ============================================================
# FEATURES
# ============================================================

RAINFALL_FEATURES = [
    "rainfall_1d",
    "rainfall_3d",
    "rainfall_7d",
    "rainfall_14d",
    "rainfall_30d",

    "rainy_days_7d",
    "rainy_days_14d",
    "rainy_days_30d",

    "dry_days_7d",
    "dry_days_14d",
    "dry_days_30d",

    "consecutive_dry_days",

    "rainfall_lag_1d",
    "rainfall_lag_3d",
    "rainfall_lag_7d",

    "rainfall_change_1d",

    "latitude",
    "longitude",
]


ENSO_FEATURES = [
    "mei_v2"
]

IOD_FEATURES = [
    "dmi"
]

MJO_FEATURES = [
    "rmm1",
    "rmm2",
    "phase",
    "amplitude",
]


# ============================================================
# EXPERIMENTS
# ============================================================

EXPERIMENTS = {
    "rainfall_only": [],

    "rainfall_plus_enso": ENSO_FEATURES,

    "rainfall_plus_iod": IOD_FEATURES,

    "rainfall_plus_enso_iod":
        ENSO_FEATURES + IOD_FEATURES,

    "rainfall_plus_mjo":
        MJO_FEATURES,

    "rainfall_plus_all_climate":
        ENSO_FEATURES
        + IOD_FEATURES
        + MJO_FEATURES,
}


# ============================================================
# TARGETS
# ============================================================

TARGETS = [
    "onset_target_7d",
    "onset_target_14d",
    "onset_target_30d",

    "break_target_7d",
    "break_target_14d",
    "break_target_30d",
]


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("CLIMATE ABLATION EXPERIMENT")
print("=" * 80)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df = df.sort_values(
    "date"
).reset_index(drop=True)


print()
print(f"Rows: {len(df):,}")


# ============================================================
# TIME SPLIT
# ============================================================

unique_dates = np.sort(
    df["date"].dropna().unique()
)

split_index = int(
    len(unique_dates) * 0.80
)

split_date = unique_dates[split_index]


train_df = df[
    df["date"] < split_date
].copy()

test_df = df[
    df["date"] >= split_date
].copy()


print()
print("TIME SPLIT")
print("-" * 80)

print(
    f"Train: "
    f"{train_df['date'].min().date()} "
    f"-> "
    f"{train_df['date'].max().date()}"
)

print(
    f"Test : "
    f"{test_df['date'].min().date()} "
    f"-> "
    f"{test_df['date'].max().date()}"
)


# ============================================================
# RUN EXPERIMENTS
# ============================================================

results = []


for experiment_name, climate_features in EXPERIMENTS.items():

    features = (
        RAINFALL_FEATURES
        + climate_features
    )

    print()
    print("=" * 80)
    print(f"EXPERIMENT: {experiment_name}")
    print(
        f"Features used: {len(features)}"
    )
    print("=" * 80)


    for target in TARGETS:

        X_train = train_df[
            features
        ]

        y_train = train_df[
            target
        ]

        X_test = test_df[
            features
        ]

        y_test = test_df[
            target
        ]


        model = RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )


        model.fit(
            X_train,
            y_train
        )


        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]


        accuracy = accuracy_score(
            y_test,
            predictions
        )

        balanced_accuracy = balanced_accuracy_score(
            y_test,
            predictions
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )


        results.append(
            {
                "experiment": experiment_name,
                "target": target,
                "feature_count": len(features),
                "accuracy": accuracy,
                "balanced_accuracy":
                    balanced_accuracy,
                "roc_auc": roc_auc,
            }
        )


        print(
            f"{target:22s} "
            f"ROC-AUC={roc_auc:.4f}"
        )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY ROC-AUC TABLE
# ============================================================

print()
print()
print("=" * 80)
print("ROC-AUC COMPARISON")
print("=" * 80)

auc_table = results_df.pivot(
    index="target",
    columns="experiment",
    values="roc_auc"
)

print(
    auc_table.round(4).to_string()
)


# ============================================================
# DISPLAY BALANCED ACCURACY
# ============================================================

print()
print("=" * 80)
print("BALANCED ACCURACY COMPARISON")
print("=" * 80)

balanced_table = results_df.pivot(
    index="target",
    columns="experiment",
    values="balanced_accuracy"
)

print(
    balanced_table.round(4).to_string()
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("ABLATION EXPERIMENT COMPLETE")
print("=" * 80)

print()
print(
    f"Results saved to:"
)

print(
    OUTPUT_FILE
)

print("=" * 80)