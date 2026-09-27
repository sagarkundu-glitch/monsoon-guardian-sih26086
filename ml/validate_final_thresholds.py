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
    "data/processed/final_threshold_test_results_2024.csv"
)


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


CLIMATE_FEATURES = {
    "rainfall": [],
    "enso": ["mei_v2"],
    "mjo": [
        "rmm1",
        "rmm2",
        "phase",
        "amplitude",
    ],
}


SELECTED = {
    "onset_target_7d": "mjo",
    "onset_target_14d": "enso",
    "onset_target_30d": "rainfall",
    "break_target_7d": "mjo",
    "break_target_14d": "enso",
    "break_target_30d": "enso",
}


# Thresholds obtained from the previous experiment.
THRESHOLDS = {
    "onset_target_7d": 0.67,
    "onset_target_14d": 0.60,
    "onset_target_30d": 0.61,
    "break_target_7d": 0.32,
    "break_target_14d": 0.27,
    "break_target_30d": 0.28,
}


print("=" * 80)
print("FINAL HELD-OUT THRESHOLD EVALUATION")
print("=" * 80)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    "date"
).reset_index(drop=True)


# ============================================================
# TEMPORAL SPLIT
# ============================================================

dates = np.sort(
    df["date"].unique()
)

n_dates = len(dates)

train_end = int(
    n_dates * 0.60
)

validation_end = int(
    n_dates * 0.80
)


train_end_date = dates[train_end]
test_start_date = dates[validation_end]


train_df = df[
    df["date"] < train_end_date
].copy()

validation_df = df[
    (df["date"] >= train_end_date)
    &
    (df["date"] < test_start_date)
].copy()

test_df = df[
    df["date"] >= test_start_date
].copy()


# ============================================================
# FINAL DEVELOPMENT DATA
# ============================================================

development_df = pd.concat(
    [
        train_df,
        validation_df,
    ],
    ignore_index=True
)


print()
print("DATA PERIODS")
print("-" * 80)

print(
    f"Training     : "
    f"{train_df.date.min().date()} -> "
    f"{train_df.date.max().date()}"
)

print(
    f"Validation   : "
    f"{validation_df.date.min().date()} -> "
    f"{validation_df.date.max().date()}"
)

print(
    f"Final test   : "
    f"{test_df.date.min().date()} -> "
    f"{test_df.date.max().date()}"
)


# ============================================================
# FINAL TEST
# ============================================================

results = []


for target, configuration in SELECTED.items():

    print()
    print("=" * 80)
    print(target)
    print(f"Configuration: {configuration}")
    print(f"Threshold: {THRESHOLDS[target]:.2f}")
    print("=" * 80)


    features = (
        RAINFALL_FEATURES
        + CLIMATE_FEATURES[configuration]
    )


    X_development = development_df[
        features
    ]

    y_development = development_df[
        target
    ]

    X_test = test_df[
        features
    ]

    y_test = test_df[
        target
    ]


    # --------------------------------------------------------
    # TRAIN ONLY ON DEVELOPMENT DATA
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )


    model.fit(
        X_development,
        y_development
    )


    # --------------------------------------------------------
    # TEST PROBABILITIES
    # --------------------------------------------------------

    probability = model.predict_proba(
        X_test
    )[:, 1]


    threshold = THRESHOLDS[target]


    prediction = (
        probability >= threshold
    ).astype(int)


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        prediction
    )

    roc_auc = roc_auc_score(
        y_test,
        probability
    )


    print(
        f"Accuracy          : {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy : {balanced_accuracy:.4f}"
    )

    print(
        f"ROC-AUC           : {roc_auc:.4f}"
    )


    results.append(
        {
            "target": target,
            "configuration": configuration,
            "threshold": threshold,
            "test_accuracy": accuracy,
            "test_balanced_accuracy":
                balanced_accuracy,
            "test_roc_auc": roc_auc,
        }
    )


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print()
print("=" * 80)
print("FINAL THRESHOLD TEST RESULTS")
print("=" * 80)

print(
    results_df.round(4).to_string(
        index=False
    )
)

print()
print(
    f"Saved to: {OUTPUT_FILE}"
)

print()
print(
    "The final test period was not used to select "
    "the thresholds."
)

print(
    "Results remain development-stage because "
    "the rainfall dataset covers only 2024."
)

print("=" * 80)