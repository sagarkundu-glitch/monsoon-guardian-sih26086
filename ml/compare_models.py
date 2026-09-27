import pandas as pd
from pathlib import Path


# ============================================================
# FILES
# ============================================================

BASELINE_FILE = Path(
    "data/processed/multihorizon_model_metrics_2024.csv"
)

CLIMATE_FILE = Path(
    "data/processed/climate_multihorizon_model_metrics_2024.csv"
)


print("=" * 80)
print("RAINFALL-ONLY vs CLIMATE-ENHANCED MODEL COMPARISON")
print("=" * 80)


# ============================================================
# LOAD
# ============================================================

baseline = pd.read_csv(BASELINE_FILE)
climate = pd.read_csv(CLIMATE_FILE)


# ============================================================
# SELECT IMPORTANT COLUMNS
# ============================================================

baseline = baseline[
    [
        "target",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
    ]
].copy()

climate = climate[
    [
        "target",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
    ]
].copy()


# ============================================================
# RENAME
# ============================================================

baseline = baseline.rename(
    columns={
        "accuracy": "baseline_accuracy",
        "balanced_accuracy": "baseline_balanced_accuracy",
        "roc_auc": "baseline_roc_auc",
    }
)

climate = climate.rename(
    columns={
        "accuracy": "climate_accuracy",
        "balanced_accuracy": "climate_balanced_accuracy",
        "roc_auc": "climate_roc_auc",
    }
)


# ============================================================
# MERGE
# ============================================================

comparison = baseline.merge(
    climate,
    on="target",
    how="inner"
)


# ============================================================
# CALCULATE IMPROVEMENT
# ============================================================

comparison["accuracy_change"] = (
    comparison["climate_accuracy"]
    - comparison["baseline_accuracy"]
)

comparison["balanced_accuracy_change"] = (
    comparison["climate_balanced_accuracy"]
    - comparison["baseline_balanced_accuracy"]
)

comparison["roc_auc_change"] = (
    comparison["climate_roc_auc"]
    - comparison["baseline_roc_auc"]
)


# ============================================================
# ROUND
# ============================================================

numeric_columns = [
    "baseline_accuracy",
    "climate_accuracy",
    "accuracy_change",

    "baseline_balanced_accuracy",
    "climate_balanced_accuracy",
    "balanced_accuracy_change",

    "baseline_roc_auc",
    "climate_roc_auc",
    "roc_auc_change",
]

comparison[numeric_columns] = comparison[
    numeric_columns
].round(4)


# ============================================================
# DISPLAY
# ============================================================

print()
print("MODEL COMPARISON")
print("-" * 80)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

output_file = Path(
    "data/processed/model_comparison_2024.csv"
)

comparison.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

for _, row in comparison.iterrows():

    target = row["target"]

    change = row["roc_auc_change"]

    if change > 0:
        result = "IMPROVED"
    elif change < 0:
        result = "DECREASED"
    else:
        result = "UNCHANGED"

    print(
        f"{target:22s} "
        f"ROC-AUC change: {change:+.4f} "
        f"-> {result}"
    )


print()
print(f"Saved comparison to:")
print(output_file)

print("=" * 80)