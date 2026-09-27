import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    roc_auc_score,
)


# ============================================================
# INPUT
# ============================================================

INPUT_FILE = Path(
    "data/processed/imd_rainfall_climate_multihorizon_2024.csv"
)

OUTPUT_FILE = Path(
    "data/processed/optimized_thresholds_2024.csv"
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


CLIMATE_FEATURES = {
    "enso": ["mei_v2"],
    "mjo": [
        "rmm1",
        "rmm2",
        "phase",
        "amplitude",
    ],
    "rainfall": [],
}


# These were selected using the validation stage.

SELECTED = {
    "onset_target_7d": "mjo",
    "onset_target_14d": "enso",
    "onset_target_30d": "rainfall",

    "break_target_7d": "mjo",
    "break_target_14d": "enso",
    "break_target_30d": "enso",
}


TARGETS = list(
    SELECTED.keys()
)


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("WARNING THRESHOLD OPTIMIZATION")
print("=" * 80)


df = pd.read_csv(
    INPUT_FILE
)

df["date"] = pd.to_datetime(
    df["date"]
)

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


train_end_date = dates[
    train_end
]

validation_end_date = dates[
    validation_end
]


train_df = df[
    df["date"] < train_end_date
].copy()


validation_df = df[
    (df["date"] >= train_end_date)
    &
    (df["date"] < validation_end_date)
].copy()


test_df = df[
    df["date"] >= validation_end_date
].copy()


# ============================================================
# COMBINE TRAIN + VALIDATION
# ============================================================

development_df = pd.concat(
    [
        train_df,
        validation_df,
    ],
    ignore_index=True
)


print()
print("DATA SPLIT")
print("-" * 80)

print(
    f"Development : "
    f"{development_df.date.min().date()} "
    f"-> "
    f"{development_df.date.max().date()}"
)

print(
    f"Validation  : "
    f"{validation_df.date.min().date()} "
    f"-> "
    f"{validation_df.date.max().date()}"
)

print(
    f"Test        : "
    f"{test_df.date.min().date()} "
    f"-> "
    f"{test_df.date.max().date()}"
)


# ============================================================
# OPTIMIZE
# ============================================================

results = []


for target, configuration in SELECTED.items():

    print()
    print("=" * 80)
    print(f"TARGET: {target}")
    print(f"CONFIGURATION: {configuration}")
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

    X_validation = validation_df[
        features
    ]

    y_validation = validation_df[
        target
    ]


    # --------------------------------------------------------
    # TRAIN MODEL
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
    # VALIDATION PROBABILITIES
    # --------------------------------------------------------

    validation_probability = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )


    validation_auc = roc_auc_score(
        y_validation,
        validation_probability
    )


    # --------------------------------------------------------
    # SEARCH THRESHOLDS
    # --------------------------------------------------------

    thresholds = np.arange(
        0.05,
        0.96,
        0.01
    )


    threshold_results = []


    for threshold in thresholds:

        prediction = (
            validation_probability
            >= threshold
        ).astype(int)


        balanced_accuracy = (
            balanced_accuracy_score(
                y_validation,
                prediction
            )
        )


        accuracy = accuracy_score(
            y_validation,
            prediction
        )


        threshold_results.append(
            {
                "threshold": threshold,
                "balanced_accuracy":
                    balanced_accuracy,
                "accuracy": accuracy,
            }
        )


    threshold_df = pd.DataFrame(
        threshold_results
    )


    # --------------------------------------------------------
    # SELECT BEST THRESHOLD
    # --------------------------------------------------------

    best_row = (
        threshold_df
        .sort_values(
            [
                "balanced_accuracy",
                "threshold",
            ],
            ascending=[
                False,
                True,
            ]
        )
        .iloc[0]
    )


    best_threshold = float(
        best_row["threshold"]
    )


    best_balanced_accuracy = float(
        best_row["balanced_accuracy"]
    )


    best_accuracy = float(
        best_row["accuracy"]
    )


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print()
    print(
        f"Validation ROC-AUC        : "
        f"{validation_auc:.4f}"
    )

    print(
        f"Optimal threshold         : "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation balanced acc.  : "
        f"{best_balanced_accuracy:.4f}"
    )

    print(
        f"Validation accuracy       : "
        f"{best_accuracy:.4f}"
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    results.append(
        {
            "target": target,
            "configuration": configuration,
            "threshold": best_threshold,
            "validation_accuracy":
                best_accuracy,
            "validation_balanced_accuracy":
                best_balanced_accuracy,
            "validation_roc_auc":
                validation_auc,
        }
    )


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(
    results
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print()
print()
print("=" * 80)
print("OPTIMIZED THRESHOLDS")
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
    "Thresholds were selected using validation data "
    "only, not the final test set."
)

print("=" * 80)