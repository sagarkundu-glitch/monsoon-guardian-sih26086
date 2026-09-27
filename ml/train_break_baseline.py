import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    classification_report,
)


INPUT_FILE = "data/processed/imd_rainfall_training_2024.csv"

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

TARGET = "break_target_7d"


def main():

    print("=" * 70)
    print("BREAK PREDICTION BASELINE TRAINING")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------
    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"\nDataset rows: {len(df):,}")
    print(f"Grid cells: {df[['latitude', 'longitude']].drop_duplicates().shape[0]}")

    # ---------------------------------------------------------
    # Check required columns
    # ---------------------------------------------------------
    required = FEATURES + [TARGET]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # ---------------------------------------------------------
    # Remove missing rows
    # ---------------------------------------------------------
    df = df.dropna(subset=required).copy()

    # ---------------------------------------------------------
    # Time-based train/test split
    # ---------------------------------------------------------
    unique_dates = np.sort(df["date"].unique())

    split_index = int(len(unique_dates) * 0.80)

    train_dates = unique_dates[:split_index]
    test_dates = unique_dates[split_index:]

    train = df[df["date"].isin(train_dates)].copy()
    test = df[df["date"].isin(test_dates)].copy()

    print("\nTIME-BASED SPLIT")
    print("-" * 70)
    print(f"Training period: {train['date'].min().date()} → {train['date'].max().date()}")
    print(f"Testing period : {test['date'].min().date()} → {test['date'].max().date()}")

    print(f"\nTraining samples: {len(train):,}")
    print(f"Testing samples : {len(test):,}")

    # ---------------------------------------------------------
    # Prepare X and y
    # ---------------------------------------------------------
    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    print("\nTARGET DISTRIBUTION")
    print("-" * 70)

    print("Training:")
    print(y_train.value_counts().sort_index())

    print("\nTesting:")
    print(y_test.value_counts().sort_index())

    # ---------------------------------------------------------
    # Train Random Forest
    # ---------------------------------------------------------
    print("\nTraining Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------
    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)

    print(f"\nAccuracy          : {accuracy:.4f}")
    print(f"Balanced Accuracy : {balanced_accuracy:.4f}")
    print(f"ROC-AUC           : {roc_auc:.4f}")

    print("\nCLASSIFICATION REPORT")
    print("-" * 70)

    print(
        classification_report(
            y_test,
            y_pred,
            digits=4,
            zero_division=0,
        )
    )

    # ---------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------
    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    print("\nFEATURE IMPORTANCE")
    print("-" * 70)

    print(importance.to_string(index=False))

    # ---------------------------------------------------------
    # Sample predictions
    # ---------------------------------------------------------
    output = test[
        ["date", "latitude", "longitude", TARGET]
    ].copy()

    output["predicted_probability"] = y_prob
    output["prediction"] = y_pred

    print("\nSAMPLE PREDICTIONS")
    print("-" * 70)

    print(
        output.head(15).to_string(index=False)
    )

    # ---------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------
    output_file = (
        "data/processed/"
        "break_baseline_predictions_2024.csv"
    )

    output.to_csv(
        output_file,
        index=False
    )

    print("\nPredictions saved to:")
    print(output_file)

    print("\n" + "=" * 70)
    print("BREAK BASELINE TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()