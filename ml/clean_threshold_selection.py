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
    "data/processed/clean_thresholds_2024.csv"
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


df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)


# ============================================================
# 60 / 20 / 20 TEMPORAL SPLIT
# ============================================================

dates = np.sort(df["date"].unique())

n = len(dates)

train_end = int(n * 0.60)
validation_end = int(n * 0.80)

train_dates = dates[:train_end]
validation_dates = dates[train_end:validation_end]
test_dates = dates[validation_end:]

train_df = df[df["date"].isin(train_dates)].copy()
validation_df = df[df["date"].isin(validation_dates)].copy()
test_df = df[df["date"].isin(test_dates)].copy()


print("=" * 80)
print("CLEAN THRESHOLD SELECTION")
print("=" * 80)

print(
    f"Train      : {train_df.date.min().date()} -> "
    f"{train_df.date.max().date()}"
)

print(
    f"Validation : {validation_df.date.min().date()} -> "
    f"{validation_df.date.max().date()}"
)

print(
    f"Test       : {test_df.date.min().date()} -> "
    f"{test_df.date.max().date()}"
)


results = []


for target, configuration in SELECTED.items():

    print()
    print("=" * 80)
    print(target)
    print(f"Configuration: {configuration}")
    print("=" * 80)


    features = (
        RAINFALL_FEATURES
        + CLIMATE_FEATURES[configuration]
    )


    # --------------------------------------------------------
    # TRAIN ONLY
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
        train_df[features],
        train_df[target]
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_probability = model.predict_proba(
        validation_df[features]
    )[:, 1]


    validation_target = validation_df[target]


    validation_auc = roc_auc_score(
        validation_target,
        validation_probability
    )


    best_threshold = 0.50
    best_balanced_accuracy = -1


    for threshold in np.arange(
        0.05,
        0.96,
        0.01
    ):

        prediction = (
            validation_probability >= threshold
        ).astype(int)


        score = balanced_accuracy_score(
            validation_target,
            prediction
        )


        if score > best_balanced_accuracy:

            best_balanced_accuracy = score
            best_threshold = threshold


    # --------------------------------------------------------
    # RETRAIN ON TRAIN + VALIDATION
    # --------------------------------------------------------

    development_df = pd.concat(
        [
            train_df,
            validation_df,
        ],
        ignore_index=True
    )


    final_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )


    final_model.fit(
        development_df[features],
        development_df[target]
    )


    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    test_probability = final_model.predict_proba(
        test_df[features]
    )[:, 1]


    test_prediction = (
        test_probability >= best_threshold
    ).astype(int)


    test_accuracy = accuracy_score(
        test_df[target],
        test_prediction
    )


    test_balanced_accuracy = balanced_accuracy_score(
        test_df[target],
        test_prediction
    )


    test_auc = roc_auc_score(
        test_df[target],
        test_probability
    )


    print(
        f"Threshold             : {best_threshold:.2f}"
    )

    print(
        f"Validation Balanced Acc: "
        f"{best_balanced_accuracy:.4f}"
    )

    print(
        f"Test Accuracy         : {test_accuracy:.4f}"
    )

    print(
        f"Test Balanced Accuracy: "
        f"{test_balanced_accuracy:.4f}"
    )

    print(
        f"Test ROC-AUC          : "
        f"{test_auc:.4f}"
    )


    results.append(
        {
            "target": target,
            "configuration": configuration,
            "threshold": best_threshold,
            "validation_balanced_accuracy":
                best_balanced_accuracy,
            "test_accuracy": test_accuracy,
            "test_balanced_accuracy":
                test_balanced_accuracy,
            "test_roc_auc": test_auc,
        }
    )


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("=" * 80)
print("CLEAN THRESHOLD SELECTION COMPLETE")
print("=" * 80)

print(
    results_df.round(4).to_string(
        index=False
    )
)

print()
print(f"Saved to: {OUTPUT_FILE}")

print()
print(
    "Thresholds were selected from TRAIN -> VALIDATION "
    "only, then evaluated once on the held-out TEST period."
)

print("=" * 80)