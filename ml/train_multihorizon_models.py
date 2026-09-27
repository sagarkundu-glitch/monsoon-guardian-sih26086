import os
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    classification_report,
)


INPUT_FILE = "data/processed/imd_rainfall_multihorizon_2024.csv"

MODEL_DIR = "data/models"

FEATURES = [
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

HORIZONS = [7, 14, 30]


def train_one_model(train, test, target):

    X_train = train[FEATURES]
    y_train = train[target]

    X_test = test[FEATURES]
    y_test = test[target]

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

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

    return (
        model,
        predictions,
        probabilities,
        accuracy,
        balanced_accuracy,
        roc_auc,
    )


def main():

    print("=" * 70)
    print("MULTI-HORIZON MODEL TRAINING")
    print("=" * 70)

    # ---------------------------------------------------------
    # Create model directory
    # ---------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    print(
        f"\nDataset rows: {len(df):,}"
    )

    print(
        "Grid cells:",
        df[["latitude", "longitude"]]
        .drop_duplicates()
        .shape[0]
    )

    # ---------------------------------------------------------
    # Check columns
    # ---------------------------------------------------------

    targets = []

    for h in HORIZONS:

        targets.append(
            f"onset_target_{h}d"
        )

        targets.append(
            f"break_target_{h}d"
        )

    required = FEATURES + targets

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    # ---------------------------------------------------------
    # Remove missing values
    # ---------------------------------------------------------

    df = df.dropna(
        subset=required
    ).copy()

    # ---------------------------------------------------------
    # Time-based split
    # ---------------------------------------------------------

    unique_dates = np.sort(
        df["date"].unique()
    )

    split_index = int(
        len(unique_dates) * 0.80
    )

    train_dates = unique_dates[
        :split_index
    ]

    test_dates = unique_dates[
        split_index:
    ]

    train = df[
        df["date"].isin(train_dates)
    ].copy()

    test = df[
        df["date"].isin(test_dates)
    ].copy()

    print("\nTIME-BASED SPLIT")
    print("-" * 70)

    print(
        f"Training: "
        f"{train['date'].min().date()} "
        f"→ "
        f"{train['date'].max().date()}"
    )

    print(
        f"Testing : "
        f"{test['date'].min().date()} "
        f"→ "
        f"{test['date'].max().date()}"
    )

    print(
        f"\nTraining samples: {len(train):,}"
    )

    print(
        f"Testing samples : {len(test):,}"
    )

    # ---------------------------------------------------------
    # Store all predictions
    # ---------------------------------------------------------

    prediction_output = test[
        [
            "date",
            "latitude",
            "longitude",
        ]
    ].copy()

    results = []

    # =========================================================
    # TRAIN ONSET MODELS
    # =========================================================

    print("\n")
    print("=" * 70)
    print("ONSET MODELS")
    print("=" * 70)

    for h in HORIZONS:

        target = f"onset_target_{h}d"

        print(
            f"\nTraining onset {h}-day model..."
        )

        (
            model,
            predictions,
            probabilities,
            accuracy,
            balanced_accuracy,
            roc_auc,
        ) = train_one_model(
            train,
            test,
            target,
        )

        model_file = (
            f"{MODEL_DIR}/"
            f"onset_{h}d_random_forest.joblib"
        )

        joblib.dump(
            model,
            model_file
        )

        prediction_output[
            f"onset_probability_{h}d"
        ] = probabilities

        prediction_output[
            f"onset_prediction_{h}d"
        ] = predictions

        results.append({
            "target": target,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "roc_auc": roc_auc,
        })

        print(
            f"Accuracy          : {accuracy:.4f}"
        )

        print(
            f"Balanced Accuracy : {balanced_accuracy:.4f}"
        )

        print(
            f"ROC-AUC           : {roc_auc:.4f}"
        )

        print(
            f"Saved model       : {model_file}"
        )

    # =========================================================
    # TRAIN BREAK MODELS
    # =========================================================

    print("\n")
    print("=" * 70)
    print("BREAK MODELS")
    print("=" * 70)

    for h in HORIZONS:

        target = f"break_target_{h}d"

        print(
            f"\nTraining break {h}-day model..."
        )

        (
            model,
            predictions,
            probabilities,
            accuracy,
            balanced_accuracy,
            roc_auc,
        ) = train_one_model(
            train,
            test,
            target,
        )

        model_file = (
            f"{MODEL_DIR}/"
            f"break_{h}d_random_forest.joblib"
        )

        joblib.dump(
            model,
            model_file
        )

        prediction_output[
            f"break_probability_{h}d"
        ] = probabilities

        prediction_output[
            f"break_prediction_{h}d"
        ] = predictions

        results.append({
            "target": target,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "roc_auc": roc_auc,
        })

        print(
            f"Accuracy          : {accuracy:.4f}"
        )

        print(
            f"Balanced Accuracy : {balanced_accuracy:.4f}"
        )

        print(
            f"ROC-AUC           : {roc_auc:.4f}"
        )

        print(
            f"Saved model       : {model_file}"
        )

    # ---------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------

    prediction_file = (
        "data/processed/"
        "multihorizon_predictions_2024.csv"
    )

    prediction_output.to_csv(
        prediction_file,
        index=False
    )

    # ---------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------

    metrics_df = pd.DataFrame(
        results
    )

    metrics_file = (
        "data/processed/"
        "multihorizon_model_metrics_2024.csv"
    )

    metrics_df.to_csv(
        metrics_file,
        index=False
    )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)

    print(
        metrics_df.to_string(
            index=False
        )
    )

    print("\nPredictions saved to:")
    print(prediction_file)

    print("\nMetrics saved to:")
    print(metrics_file)

    print("\nModels saved in:")
    print(MODEL_DIR)

    print("\n")
    print("=" * 70)
    print("MULTI-HORIZON MODEL TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()