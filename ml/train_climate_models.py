import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)

import joblib


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    "data/processed/imd_rainfall_climate_multihorizon_2024.csv"
)

MODEL_DIR = Path("data/models")

PREDICTION_FILE = Path(
    "data/processed/climate_multihorizon_predictions_2024.csv"
)

METRICS_FILE = Path(
    "data/processed/climate_multihorizon_model_metrics_2024.csv"
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


# Climate predictors

CLIMATE_FEATURES = [
    "mei_v2",
    "dmi",
    "rmm1",
    "rmm2",
    "phase",
    "amplitude",
]


FEATURES = RAINFALL_FEATURES + CLIMATE_FEATURES


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
# LOAD DATA
# ============================================================

print("=" * 70)
print("CLIMATE-ENHANCED MULTIHORIZON MODEL TRAINING")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )


df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


print()
print(f"Rows loaded    : {len(df):,}")
print(f"Columns loaded : {len(df.columns)}")


# ============================================================
# VERIFY FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing feature columns:\n"
        + "\n".join(missing_features)
    )


# ============================================================
# VERIFY TARGETS
# ============================================================

missing_targets = [
    target
    for target in TARGETS
    if target not in df.columns
]

if missing_targets:

    raise ValueError(
        "Missing target columns:\n"
        + "\n".join(missing_targets)
    )


print()
print(f"✓ All {len(FEATURES)} features found")
print(f"✓ All {len(TARGETS)} targets found")


# ============================================================
# CHECK MISSING FEATURES
# ============================================================

missing_values = df[FEATURES].isna().sum()

total_missing = missing_values.sum()

print()
print(
    f"Total missing feature values: {total_missing:,}"
)

if total_missing > 0:

    print()
    print("Missing values:")

    print(
        missing_values[
            missing_values > 0
        ]
    )

    raise ValueError(
        "Missing feature values detected."
    )


# ============================================================
# SORT BY DATE
# ============================================================

df = df.sort_values(
    "date"
).reset_index(drop=True)


# ============================================================
# TIME-BASED TRAIN / TEST SPLIT
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
print("=" * 70)
print("TIME-BASED TRAIN / TEST SPLIT")
print("=" * 70)

print(
    f"Split date     : {pd.Timestamp(split_date).date()}"
)

print(
    f"Training dates : "
    f"{train_df['date'].min().date()} "
    f"-> "
    f"{train_df['date'].max().date()}"
)

print(
    f"Testing dates  : "
    f"{test_df['date'].min().date()} "
    f"-> "
    f"{test_df['date'].max().date()}"
)

print(
    f"Training rows  : {len(train_df):,}"
)

print(
    f"Testing rows   : {len(test_df):,}"
)


# ============================================================
# MODEL DIRECTORY
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PREDICTION DATAFRAME
# ============================================================

all_predictions = test_df[
    [
        "date",
        "latitude",
        "longitude",
    ]
].copy()


# ============================================================
# METRICS
# ============================================================

all_metrics = []


# ============================================================
# TRAIN SIX MODELS
# ============================================================

for target in TARGETS:

    print()
    print()
    print("=" * 70)
    print(f"TRAINING MODEL: {target}")
    print("=" * 70)


    # --------------------------------------------------------
    # TRAIN / TEST DATA
    # --------------------------------------------------------

    X_train = train_df[
        FEATURES
    ]

    y_train = train_df[
        target
    ]

    X_test = test_df[
        FEATURES
    ]

    y_test = test_df[
        target
    ]


    print(
        f"Training samples       : {len(X_train):,}"
    )

    print(
        f"Testing samples        : {len(X_test):,}"
    )

    print(
        f"Positive train samples : {int(y_train.sum()):,}"
    )

    print(
        f"Positive test samples  : {int(y_test.sum()):,}"
    )


    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print()
    print("MODEL RESULTS")
    print("-" * 70)

    print(
        f"Accuracy          : {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy : {balanced_accuracy:.4f}"
    )

    print(
        f"ROC-AUC           : {roc_auc:.4f}"
    )


    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_name = (
        target.replace(
            "_target",
            ""
        )
        + "_climate_random_forest.joblib"
    )

    model_path = (
        MODEL_DIR / model_name
    )


    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "target": target,
            "model_type": "RandomForestClassifier",
        },
        model_path
    )


    print()
    print(
        f"Model saved: {model_path}"
    )


    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    probability_column = (
        target
        + "_climate_probability"
    )

    prediction_column = (
        target
        + "_climate_prediction"
    )


    all_predictions[
        probability_column
    ] = probabilities

    all_predictions[
        prediction_column
    ] = predictions


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False
    )


    print()
    print("TOP 10 FEATURES")
    print("-" * 70)

    print(
        importance_df.head(10).to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    all_metrics.append(
        {
            "target": target,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "roc_auc": roc_auc,
            "training_rows": len(X_train),
            "testing_rows": len(X_test),
            "feature_count": len(FEATURES),
        }
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

all_predictions.to_csv(
    PREDICTION_FILE,
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame(
    all_metrics
)

metrics_df.to_csv(
    METRICS_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print()
print("=" * 70)
print("CLIMATE-ENHANCED TRAINING COMPLETE")
print("=" * 70)

print()
print("FINAL MODEL PERFORMANCE")
print("-" * 70)

print(
    metrics_df.to_string(
        index=False
    )
)

print()
print(
    f"Predictions saved to:"
)

print(
    PREDICTION_FILE
)

print()
print(
    f"Metrics saved to:"
)

print(
    METRICS_FILE
)

print()
print(
    "Models saved in:"
)

print(
    MODEL_DIR
)

print()
print(
    f"Total features used: {len(FEATURES)}"
)

print("=" * 70)