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

MODEL_DIR = Path("data/models/final_candidates")

OUTPUT_FILE = Path(
    "data/processed/final_candidate_model_metrics_2024.csv"
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
    "iod": ["dmi"],
    "mjo": ["rmm1", "rmm2", "phase", "amplitude"],
}


# ============================================================
# SELECTED CONFIGURATIONS
# ============================================================

SELECTED = {
    "onset_target_7d": "mjo",
    "onset_target_14d": "enso",
    "onset_target_30d": "rainfall",

    "break_target_7d": "mjo",
    "break_target_14d": "enso",
    "break_target_30d": "enso",
}


TARGETS = list(SELECTED.keys())


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("TRAINING SELECTED CANDIDATE MODELS")
print("=" * 80)

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    "date"
).reset_index(drop=True)


print()
print(f"Rows loaded: {len(df):,}")


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
test_start_date = dates[validation_end]


train_df = df[
    df["date"] < train_end_date
].copy()

validation_df = df[
    (df["date"] >= train_end_date) &
    (df["date"] < test_start_date)
].copy()

test_df = df[
    df["date"] >= test_start_date
].copy()


# Combine train + validation for final candidate training

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
    f"Development: "
    f"{development_df['date'].min().date()} -> "
    f"{development_df['date'].max().date()}"
)

print(
    f"Final test: "
    f"{test_df['date'].min().date()} -> "
    f"{test_df['date'].max().date()}"
)

print(
    f"Development rows: {len(development_df):,}"
)

print(
    f"Test rows: {len(test_df):,}"
)


# ============================================================
# MODEL DIRECTORY
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TRAIN
# ============================================================

results = []


for target, configuration in SELECTED.items():

    print()
    print("=" * 80)
    print(f"TARGET: {target}")
    print(f"CONFIGURATION: {configuration}")
    print("=" * 80)


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    features = RAINFALL_FEATURES.copy()

    if configuration != "rainfall":

        features += CLIMATE_FEATURES[
            configuration
        ]


    print()
    print(
        f"Features used: {len(features)}"
    )

    print(
        ", ".join(features)
    )


    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

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
    # MODEL
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
    # TEST
    # --------------------------------------------------------

    test_predictions = model.predict(
        X_test
    )

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]


    accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        test_predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        test_probabilities
    )


    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print()
    print("HELD-OUT TEST PERFORMANCE")
    print("-" * 80)

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

    model_filename = (
        target
        + "_selected_random_forest.joblib"
    )

    model_path = (
        MODEL_DIR / model_filename
    )


    joblib.dump(
        {
            "model": model,
            "features": features,
            "target": target,
            "configuration": configuration,
            "model_type": "RandomForestClassifier",
            "training_period": (
                f"{development_df['date'].min().date()}"
                f"_to_"
                f"{development_df['date'].max().date()}"
            ),
        },
        model_path
    )


    print(
        f"Saved: {model_path}"
    )


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance":
                model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False
    )


    print()
    print("TOP FEATURES")
    print("-" * 80)

    print(
        importance.head(8).to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results.append(
        {
            "target": target,
            "configuration": configuration,
            "feature_count": len(features),
            "test_accuracy": accuracy,
            "test_balanced_accuracy":
                balanced_accuracy,
            "test_roc_auc": roc_auc,
        }
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print()
print("=" * 80)
print("SELECTED CANDIDATE MODELS COMPLETE")
print("=" * 80)

print()
print(
    results_df.round(4).to_string(
        index=False
    )
)

print()
print(
    f"Metrics saved to: {OUTPUT_FILE}"
)

print()
print(
    f"Models saved to: {MODEL_DIR}"
)

print()
print(
    "These are prototype candidate models based on "
    "2024 data and should not yet be described as "
    "operational forecasting models."
)

print("=" * 80)