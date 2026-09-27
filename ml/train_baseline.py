from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
)


INPUT_FILE = Path(
    "data/processed/imd_rainfall_training_2024.csv"
)


def main():

    print("=" * 70)
    print("SIH26086 — BASELINE ML MODEL")
    print("=" * 70)

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found:\n{INPUT_FILE}"
        )

    print("\nLoading training dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"]
    )

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    print(
        f"Rows: {len(df):,}"
    )

    # -----------------------------------------------------
    # FEATURES
    # -----------------------------------------------------

    feature_columns = [
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

    # -----------------------------------------------------
    # TARGET
    # -----------------------------------------------------

    target_column = "onset_target_7d"

    X = df[feature_columns]

    y = df[target_column]

    # -----------------------------------------------------
    # TIME-BASED SPLIT
    # -----------------------------------------------------

    # Use the first 80% of dates for training and the
    # final 20% for testing.

    unique_dates = sorted(
        df["date"].unique()
    )

    split_position = int(
        len(unique_dates) * 0.80
    )

    split_date = unique_dates[
        split_position
    ]

    train_mask = (
        df["date"] < split_date
    )

    test_mask = (
        df["date"] >= split_date
    )

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    print("\n" + "=" * 70)
    print("TIME-BASED SPLIT")
    print("=" * 70)

    print(
        f"Training dates: "
        f"{df.loc[train_mask, 'date'].min().date()} "
        f"→ "
        f"{df.loc[train_mask, 'date'].max().date()}"
    )

    print(
        f"Testing dates : "
        f"{df.loc[test_mask, 'date'].min().date()} "
        f"→ "
        f"{df.loc[test_mask, 'date'].max().date()}"
    )

    print(
        f"\nTraining rows: {len(X_train):,}"
    )

    print(
        f"Testing rows : {len(X_test):,}"
    )

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    print("\nTraining Random Forest baseline...")

    model = RandomForestClassifier(
        n_estimators=300,
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

    print(
        "Training complete."
    )

    # -----------------------------------------------------
    # PREDICTIONS
    # -----------------------------------------------------

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # -----------------------------------------------------
    # EVALUATION
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            predictions
        )
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)

    print(
        f"Accuracy          : {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy : {balanced_accuracy:.4f}"
    )

    print(
        f"ROC-AUC           : {auc:.4f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            digits=4
        )
    )

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    print(
        importance.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # SAMPLE PROBABILITIES
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("SAMPLE PROBABILISTIC OUTPUT")
    print("=" * 70)

    sample = df.loc[
        test_mask,
        [
            "date",
            "latitude",
            "longitude",
            "onset_target_7d",
        ]
    ].copy()

    sample["predicted_probability"] = (
        probabilities
    )

    print(
        sample.head(20).to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()