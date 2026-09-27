import pandas as pd
import joblib

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data/processed/imd_rainfall_climate_multihorizon_2024.csv"
)

MODEL_DIR = PROJECT_ROOT / "data/models/final_candidates"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_CONFIG = {
    "onset_target_7d": {
        "file": "onset_target_7d_selected_random_forest.joblib",
        "threshold": 0.79,
    },

    "onset_target_14d": {
        "file": "onset_target_14d_selected_random_forest.joblib",
        "threshold": 0.87,
    },

    "onset_target_30d": {
        "file": "onset_target_30d_selected_random_forest.joblib",
        "threshold": 0.90,
    },

    "break_target_7d": {
        "file": "break_target_7d_selected_random_forest.joblib",
        "threshold": 0.20,
    },

    "break_target_14d": {
        "file": "break_target_14d_selected_random_forest.joblib",
        "threshold": 0.08,
    },

    "break_target_30d": {
        "file": "break_target_30d_selected_random_forest.joblib",
        "threshold": 0.05,
    },
}


# ============================================================
# PREDICTION ENGINE
# ============================================================

class MonsoonPredictionEngine:

    def __init__(
        self,
        data_file=DATA_FILE,
        model_dir=MODEL_DIR,
    ):

        self.data_file = Path(
            data_file
        )

        self.model_dir = Path(
            model_dir
        )

        # ----------------------------------------------------
        # LOAD DATA
        # ----------------------------------------------------

        if not self.data_file.exists():

            raise FileNotFoundError(
                f"Prediction data not found: "
                f"{self.data_file}"
            )


        self.data = pd.read_csv(
            self.data_file
        )


        self.data["date"] = pd.to_datetime(
            self.data["date"]
        )


        # ----------------------------------------------------
        # CHECK MODEL FILES
        # ----------------------------------------------------

        # These model files are large. Keep their paths here and
        # load one model at a time during inference to reduce RAM use
        # on small hosting plans.
        self.model_paths = {}

        for target, config in MODEL_CONFIG.items():

            model_path = self.model_dir / config["file"]

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found: {model_path}"
                )

            self.model_paths[target] = model_path


        print(
            "Prediction engine initialized"
        )

        print(
            f"Data rows: "
            f"{len(self.data):,}"
        )

        print(
            f"Model files available: "
            f"{len(self.model_paths)}"
        )


    # ========================================================
    # AVAILABLE LOCATIONS
    # ========================================================

    def get_locations(self):

        locations = (
            self.data[
                [
                    "latitude",
                    "longitude",
                ]
            ]
            .drop_duplicates()
            .sort_values(
                [
                    "latitude",
                    "longitude",
                ]
            )
            .reset_index(drop=True)
        )


        return locations


    # ========================================================
    # FIND NEAREST GRID CELL
    # ========================================================

    def find_nearest_location(
        self,
        latitude,
        longitude,
    ):

        locations = self.get_locations()


        locations["distance"] = (
            (locations["latitude"] - latitude) ** 2
            +
            (locations["longitude"] - longitude) ** 2
        )


        nearest = locations.loc[
            locations["distance"].idxmin()
        ]


        return {
            "latitude": float(
                nearest["latitude"]
            ),
            "longitude": float(
                nearest["longitude"]
            ),
        }


    # ========================================================
    # GET FEATURE ROW
    # ========================================================

    def get_feature_row(
        self,
        latitude,
        longitude,
        date,
    ):

        date = pd.Timestamp(date)


        # ----------------------------------------------------
        # Find nearest available grid cell
        # ----------------------------------------------------

        nearest = self.find_nearest_location(
            latitude,
            longitude,
        )


        lat = nearest["latitude"]
        lon = nearest["longitude"]


        # ----------------------------------------------------
        # Exact date + grid
        # ----------------------------------------------------

        rows = self.data[
            (self.data["date"] == date)
            &
            (self.data["latitude"] == lat)
            &
            (self.data["longitude"] == lon)
        ]


        if rows.empty:

            raise ValueError(
                "No feature data available for "
                f"{date.date()} at grid "
                f"({lat}, {lon}). "
                "Try another date between "
                "2024-01-30 and 2024-12-01."
            )


        return rows.iloc[0], nearest


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        latitude,
        longitude,
        date,
    ):

        row, nearest = self.get_feature_row(
            latitude,
            longitude,
            date,
        )


        result = {

            "requested_location": {
                "latitude": float(latitude),
                "longitude": float(longitude),
            },

            "grid_location": {
                "latitude": nearest["latitude"],
                "longitude": nearest["longitude"],
            },

            "date": str(
                pd.Timestamp(date).date()
            ),

            "onset": {},

            "break": {},
        }


        # ====================================================
        # ONSET + BREAK PREDICTIONS
        # ====================================================

        for target, model_path in self.model_paths.items():

            bundle = joblib.load(model_path)
            model = bundle["model"]
            # Avoid spawning worker processes for a single-row prediction.
            # This is more reliable on Windows and small hosted instances.
            if hasattr(model, "set_params"):
                model.set_params(n_jobs=1)

            features = bundle["features"]


            # ------------------------------------------------
            # Make feature vector
            # ------------------------------------------------

            X = pd.DataFrame(
                [
                    [
                        row[feature]
                        for feature in features
                    ]
                ],
                columns=features,
            )


            # ------------------------------------------------
            # Probability
            # ------------------------------------------------

            probability = model.predict_proba(
                X
            )[0][1]


            probability = float(
                probability
            )


            threshold = float(
                MODEL_CONFIG[target][
                    "threshold"
                ]
            )


            signal = (
                probability >= threshold
            )


            # ------------------------------------------------
            # Horizon
            # ------------------------------------------------

            if "7d" in target:

                horizon = "7_day"

            elif "14d" in target:

                horizon = "14_day"

            elif "30d" in target:

                horizon = "30_day"

            else:

                horizon = "unknown"


            # ------------------------------------------------
            # Result
            # ------------------------------------------------

            prediction = {

                "probability": round(
                    probability,
                    4,
                ),

                "threshold": round(
                    threshold,
                    2,
                ),

                "prototype_signal": bool(
                    signal
                ),
            }


            # ------------------------------------------------
            # Store
            # ------------------------------------------------

            if target.startswith(
                "onset"
            ):

                result[
                    "onset"
                ][horizon] = prediction


            elif target.startswith(
                "break"
            ):

                result[
                    "break"
                ][horizon] = prediction

            # Release each forest before loading the next one.
            del X, model, bundle


        # ====================================================
        # MODEL DISCLAIMER
        # ====================================================

        result["model_information"] = {

            "dataset": "IMD rainfall + climate predictors",

            "training_period": "2024",

            "prediction_type":
                "development-stage prototype",

            "note":
                "Probabilities and prototype signals are "
                "experimental and are not official IMD "
                "monsoon onset or break declarations.",
        }


        return result


# ============================================================
# TEST ENGINE DIRECTLY
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("MONSOON GUARDIAN PREDICTION ENGINE TEST")
    print("=" * 70)


    engine = MonsoonPredictionEngine()


    # --------------------------------------------------------
    # Show available locations
    # --------------------------------------------------------

    locations = engine.get_locations()


    print()
    print(
        f"Available grid cells: "
        f"{len(locations)}"
    )


    print()
    print("First 5 grid cells:")

    print(
        locations.head(5).to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Test prediction
    # --------------------------------------------------------

    first_location = locations.iloc[0]


    latitude = float(
        first_location["latitude"]
    )

    longitude = float(
        first_location["longitude"]
    )


    print()
    print(
        "Testing prediction at:"
    )

    print(
        f"Latitude : {latitude}"
    )

    print(
        f"Longitude: {longitude}"
    )

    print(
        "Date     : 2024-10-01"
    )


    result = engine.predict(
        latitude=latitude,
        longitude=longitude,
        date="2024-10-01",
    )


    # --------------------------------------------------------
    # Pretty print result
    # --------------------------------------------------------

    import json


    print()
    print("=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)

    print(
        json.dumps(
            result,
            indent=2
        )
    )

    print()
    print("=" * 70)
    print("PREDICTION ENGINE TEST COMPLETE")
    print("=" * 70)
