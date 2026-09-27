import pandas as pd
import joblib
from pathlib import Path


MODEL_DIR = Path("data/models")

OUTPUT_FILE = Path(
    "data/processed/climate_feature_importance_2024.csv"
)


MODELS = [
    "onset_7d_climate_random_forest.joblib",
    "onset_14d_climate_random_forest.joblib",
    "onset_30d_climate_random_forest.joblib",
    "break_7d_climate_random_forest.joblib",
    "break_14d_climate_random_forest.joblib",
    "break_30d_climate_random_forest.joblib",
]


print("=" * 80)
print("CLIMATE FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)


all_results = []


for model_file in MODELS:

    model_path = MODEL_DIR / model_file

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    bundle = joblib.load(model_path)

    model = bundle["model"]
    features = bundle["features"]
    target = bundle["target"]


    importance = pd.DataFrame(
        {
            "target": target,
            "feature": features,
            "importance": model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False
    ).reset_index(drop=True)

    importance["rank"] = (
        importance.index + 1
    )

    all_results.append(importance)


    print()
    print("=" * 80)
    print(target)
    print("=" * 80)

    print(
        importance[
            [
                "rank",
                "feature",
                "importance",
            ]
        ].head(15).to_string(index=False)
    )


# ============================================================
# COMBINE
# ============================================================

results = pd.concat(
    all_results,
    ignore_index=True
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# CLIMATE-ONLY SUMMARY
# ============================================================

climate_features = [
    "mei_v2",
    "dmi",
    "rmm1",
    "rmm2",
    "phase",
    "amplitude",
]


climate_results = results[
    results["feature"].isin(
        climate_features
    )
].copy()


print()
print("=" * 80)
print("CLIMATE FEATURE IMPORTANCE")
print("=" * 80)


climate_summary = (
    climate_results
    .groupby("feature")["importance"]
    .agg(
        mean_importance="mean",
        max_importance="max",
        min_importance="min",
    )
    .sort_values(
        "mean_importance",
        ascending=False
    )
)


print(
    climate_summary.to_string()
)


print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print()
print(
    f"Saved to: {OUTPUT_FILE}"
)