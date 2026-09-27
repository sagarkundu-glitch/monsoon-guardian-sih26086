import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)


INPUT_FILE = Path(
    "data/processed/imd_rainfall_climate_multihorizon_2024.csv"
)

OUTPUT_FILE = Path(
    "data/processed/robust_model_selection_2024.csv"
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

ENSO = ["mei_v2"]
IOD = ["dmi"]
MJO = ["rmm1", "rmm2", "phase", "amplitude"]


EXPERIMENTS = {
    "rainfall_only": [],

    "rainfall_plus_enso":
        ENSO,

    "rainfall_plus_iod":
        IOD,

    "rainfall_plus_enso_iod":
        ENSO + IOD,

    "rainfall_plus_mjo":
        MJO,

    "rainfall_plus_all_climate":
        ENSO + IOD + MJO,
}


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
print("ROBUST TEMPORAL MODEL SELECTION")
print("=" * 80)

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)


print()
print(f"Rows: {len(df):,}")


# ============================================================
# TEMPORAL SPLIT
# ============================================================

dates = np.sort(
    df["date"].unique()
)

n_dates = len(dates)

train_end = int(n_dates * 0.60)
validation_end = int(n_dates * 0.80)

train_end_date = dates[train_end]
validation_end_date = dates[validation_end]


train_df = df[
    df["date"] < train_end_date
].copy()

validation_df = df[
    (df["date"] >= train_end_date) &
    (df["date"] < validation_end_date)
].copy()

test_df = df[
    df["date"] >= validation_end_date
].copy()


print()
print("=" * 80)
print("TEMPORAL SPLIT")
print("=" * 80)

print(
    f"TRAIN      : "
    f"{train_df.date.min().date()} -> "
    f"{train_df.date.max().date()}"
)

print(
    f"VALIDATION : "
    f"{validation_df.date.min().date()} -> "
    f"{validation_df.date.max().date()}"
)

print(
    f"TEST       : "
    f"{test_df.date.min().date()} -> "
    f"{test_df.date.max().date()}"
)

print()
print(f"Train rows      : {len(train_df):,}")
print(f"Validation rows : {len(validation_df):,}")
print(f"Test rows       : {len(test_df):,}")


# ============================================================
# EXPERIMENTS
# ============================================================

validation_results = []


for experiment_name, climate_features in EXPERIMENTS.items():

    features = (
        RAINFALL_FEATURES
        + climate_features
    )

    print()
    print("-" * 80)
    print(
        f"EXPERIMENT: {experiment_name}"
    )

    for target in TARGETS:

        X_train = train_df[features]
        y_train = train_df[target]

        X_val = validation_df[features]
        y_val = validation_df[target]


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


        val_pred = model.predict(
            X_val
        )

        val_prob = model.predict_proba(
            X_val
        )[:, 1]


        accuracy = accuracy_score(
            y_val,
            val_pred
        )

        balanced_accuracy = balanced_accuracy_score(
            y_val,
            val_pred
        )

        roc_auc = roc_auc_score(
            y_val,
            val_prob
        )


        validation_results.append(
            {
                "experiment": experiment_name,
                "target": target,
                "feature_count": len(features),
                "validation_accuracy": accuracy,
                "validation_balanced_accuracy":
                    balanced_accuracy,
                "validation_roc_auc": roc_auc,
            }
        )


results = pd.DataFrame(
    validation_results
)


# ============================================================
# SELECT USING VALIDATION ONLY
# ============================================================

selected = (
    results.sort_values(
        [
            "target",
            "validation_roc_auc",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .groupby(
        "target",
        as_index=False
    )
    .first()
)


# ============================================================
# TEST SELECTED CONFIGURATIONS
# ============================================================

final_results = []


for _, row in selected.iterrows():

    target = row["target"]

    experiment = row["experiment"]

    climate_features = EXPERIMENTS[
        experiment
    ]

    features = (
        RAINFALL_FEATURES
        + climate_features
    )


    X_train = train_df[features]
    y_train = train_df[target]

    X_val = validation_df[features]
    y_val = validation_df[target]

    X_test = test_df[features]
    y_test = test_df[target]


    # Train using TRAIN + VALIDATION
    combined_train = pd.concat(
        [
            train_df,
            validation_df,
        ],
        ignore_index=True
    )


    X_combined = combined_train[
        features
    ]

    y_combined = combined_train[
        target
    ]


    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )


    model.fit(
        X_combined,
        y_combined
    )


    test_pred = model.predict(
        X_test
    )

    test_prob = model.predict_proba(
        X_test
    )[:, 1]


    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )

    test_balanced_accuracy = balanced_accuracy_score(
        y_test,
        test_pred
    )

    test_roc_auc = roc_auc_score(
        y_test,
        test_prob
    )


    final_results.append(
        {
            "target": target,
            "selected_experiment": experiment,
            "feature_count": len(features),

            "validation_roc_auc":
                row["validation_roc_auc"],

            "validation_balanced_accuracy":
                row["validation_balanced_accuracy"],

            "test_accuracy":
                test_accuracy,

            "test_balanced_accuracy":
                test_balanced_accuracy,

            "test_roc_auc":
                test_roc_auc,
        }
    )


# ============================================================
# FINAL RESULTS
# ============================================================

final_df = pd.DataFrame(
    final_results
)


print()
print()
print("=" * 80)
print("SELECTED CONFIGURATIONS")
print("=" * 80)

print(
    selected[
        [
            "target",
            "experiment",
            "feature_count",
            "validation_balanced_accuracy",
            "validation_roc_auc",
        ]
    ]
    .round(4)
    .to_string(index=False)
)


print()
print("=" * 80)
print("FINAL HELD-OUT TEST RESULTS")
print("=" * 80)

print(
    final_df.round(4).to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("=" * 80)
print("ROBUST MODEL SELECTION COMPLETE")
print("=" * 80)

print()
print(
    f"Saved to: {OUTPUT_FILE}"
)

print()
print(
    "The test set was NOT used to select the "
    "climate configuration."
)

print(
    "This is a development-stage evaluation using "
    "one year of data."
)

print("=" * 80)