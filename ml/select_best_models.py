import pandas as pd
from pathlib import Path


# ============================================================
# FILES
# ============================================================

INPUT_FILE = Path(
    "data/processed/climate_ablation_results_2024.csv"
)

OUTPUT_FILE = Path(
    "data/processed/selected_model_configurations_2024.csv"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("MODEL CONFIGURATION SELECTION")
print("=" * 80)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"File not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)


# ============================================================
# VALIDATE
# ============================================================

required_columns = [
    "experiment",
    "target",
    "feature_count",
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        "Missing columns:\n"
        + "\n".join(missing)
    )


# ============================================================
# ROUND FOR DISPLAY
# ============================================================

display_df = df.copy()

display_df["accuracy"] = display_df[
    "accuracy"
].round(4)

display_df["balanced_accuracy"] = display_df[
    "balanced_accuracy"
].round(4)

display_df["roc_auc"] = display_df[
    "roc_auc"
].round(4)


# ============================================================
# BEST BY ROC-AUC
# ============================================================

best_auc = (
    df.sort_values(
        ["target", "roc_auc"],
        ascending=[True, False]
    )
    .groupby(
        "target",
        as_index=False
    )
    .first()
)


# ============================================================
# BEST BY BALANCED ACCURACY
# ============================================================

best_balanced = (
    df.sort_values(
        ["target", "balanced_accuracy"],
        ascending=[True, False]
    )
    .groupby(
        "target",
        as_index=False
    )
    .first()
)


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 80)
print("BEST CONFIGURATION BY ROC-AUC")
print("=" * 80)

print(
    best_auc[
        [
            "target",
            "experiment",
            "feature_count",
            "accuracy",
            "balanced_accuracy",
            "roc_auc",
        ]
    ]
    .round(4)
    .to_string(index=False)
)


print()
print("=" * 80)
print("BEST CONFIGURATION BY BALANCED ACCURACY")
print("=" * 80)

print(
    best_balanced[
        [
            "target",
            "experiment",
            "feature_count",
            "accuracy",
            "balanced_accuracy",
            "roc_auc",
        ]
    ]
    .round(4)
    .to_string(index=False)
)


# ============================================================
# SAVE ROC-AUC SELECTION
# ============================================================

best_auc.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("MODEL SELECTION COMPLETE")
print("=" * 80)

print()
print(
    f"Saved to: {OUTPUT_FILE}"
)

print()
print(
    "IMPORTANT:"
)

print(
    "These selections are development-stage results "
    "from the 2024 prototype dataset."
)

print(
    "They should not yet be presented as operational "
    "forecast skill."
)

print("=" * 80)