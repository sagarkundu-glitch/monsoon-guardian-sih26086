from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import sys


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ML_DIRECTORY = PROJECT_ROOT / "ml"
FRONTEND_DIRECTORY = PROJECT_ROOT / "frontend"

sys.path.insert(0, str(ML_DIRECTORY))


# ============================================================
# IMPORT PREDICTION ENGINE
# ============================================================

from prediction_engine import MonsoonPredictionEngine


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Monsoon Guardian API",
    description=(
        "Hyperlocal monsoon onset and break "
        "prediction prototype for SIH 2026."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD PREDICTION ENGINE
# ============================================================

try:
    engine = MonsoonPredictionEngine()
    ENGINE_STATUS = "ready"

except Exception as error:
    engine = None
    ENGINE_STATUS = "unavailable"
    print(f"Prediction engine failed to initialize: {error}")


# ============================================================
# REQUEST MODEL
# ============================================================

class PredictionRequest(BaseModel):

    latitude: float = Field(
        ...,
        description="Requested latitude",
        ge=-90,
        le=90,
    )

    longitude: float = Field(
        ...,
        description="Requested longitude",
        ge=-180,
        le=180,
    )

    date: str = Field(
        ...,
        description="Prediction date in YYYY-MM-DD format",
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return FileResponse(FRONTEND_DIRECTORY / "index.html")


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok" if engine is not None else "error",
        "engine": ENGINE_STATUS,
    }


# ============================================================
# LOCATIONS ENDPOINT
# ============================================================

@app.get("/locations")
def locations():

    if engine is None:

        raise HTTPException(
            status_code=500,
            detail="Prediction engine is not available.",
        )

    location_df = engine.get_locations()

    locations = []

    for _, row in location_df.iterrows():

        locations.append(
            {
                "latitude": float(
                    row["latitude"]
                ),
                "longitude": float(
                    row["longitude"]
                ),
            }
        )

    return {
        "count": len(locations),
        "locations": locations,
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    if engine is None:

        raise HTTPException(
            status_code=500,
            detail="Prediction engine is not available.",
        )

    try:

        result = engine.predict(
            latitude=request.latitude,
            longitude=request.longitude,
            date=request.date,
        )

        return result

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ============================================================
# SYSTEM INFORMATION
# ============================================================

@app.get("/info")
def info():

    return {
        "project": "Monsoon Guardian",

        "problem_statement": "SIH26086",

        "dataset": (
            "IMD 0.25 degree rainfall "
            "+ climate predictors"
        ),

        "climate_predictors": [
            "ENSO / MEI",
            "IOD / DMI",
            "MJO / RMM1",
            "MJO / RMM2",
            "MJO Phase",
            "MJO Amplitude",
        ],

        "prediction_horizons": [
            "7 days",
            "14 days",
            "30 days",
        ],

        "prediction_types": [
            "Monsoon onset",
            "Monsoon break",
        ],

        "training_period": "2024",

        "system_status": (
            "development-stage prototype"
        ),

        "important_note": (
            "This system is experimental and does "
            "not issue official IMD declarations."
        ),
    }


# Serve the dashboard files from the same public URL as the API.
# Keep this mount after the API routes so it does not intercept them.
app.mount(
    "/",
    StaticFiles(directory=str(FRONTEND_DIRECTORY)),
    name="frontend",
)
