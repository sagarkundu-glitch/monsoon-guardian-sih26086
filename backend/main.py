from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import sys
import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


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

_WEATHER_CACHE = {}
_WEATHER_CACHE_SECONDS = 600


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


@app.get("/weather/live")
def live_weather(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
):
    """Return a separate live-weather snapshot for the selected map point."""
    # Match the dashboard's displayed precision and avoid sending unnecessary
    # coordinate precision to the weather provider.
    latitude = round(latitude, 2)
    longitude = round(longitude, 2)
    cache_key = (latitude, longitude)
    cached = _WEATHER_CACHE.get(cache_key)
    now = time.monotonic()
    if cached and now - cached[0] < _WEATHER_CACHE_SECONDS:
        return cached[1]

    params = urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation,rain,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum",
        "forecast_days": 1,
        "timezone": "auto",
        "wind_speed_unit": "kmh",
    })
    request = Request(
        f"https://api.open-meteo.com/v1/forecast?{params}",
        headers={
            "Accept": "application/json",
            "User-Agent": "MonsoonGuardian/1.0",
        },
    )

    try:
        with urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        print(f"Live weather request failed: {error}")
        raise HTTPException(
            status_code=502,
            detail="Live weather is temporarily unavailable. Please try again shortly.",
        ) from error

    if not isinstance(payload.get("current"), dict) or not isinstance(payload.get("daily"), dict):
        raise HTTPException(
            status_code=502,
            detail="The weather provider returned incomplete data. Please try again shortly.",
        )

    current = payload.get("current", {})
    daily = payload.get("daily", {})

    def value(source, key, index=None):
        result = source.get(key)
        if index is not None:
            result = result[index] if isinstance(result, list) and len(result) > index else None
        return result

    result = {
        "source": "Open-Meteo",
        "source_url": "https://open-meteo.com/",
        "timezone": payload.get("timezone", "UTC"),
        "observed_at": current.get("time"),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "location": {
            "latitude": payload.get("latitude", latitude),
            "longitude": payload.get("longitude", longitude),
        },
        "current": {
            "temperature_c": value(current, "temperature_2m"),
            "relative_humidity_pct": value(current, "relative_humidity_2m"),
            "precipitation_mm": value(current, "precipitation"),
            "rain_mm": value(current, "rain"),
            "wind_speed_kmh": value(current, "wind_speed_10m"),
        },
        "today": {
            "date": value(daily, "time", 0),
            "temperature_max_c": value(daily, "temperature_2m_max", 0),
            "temperature_min_c": value(daily, "temperature_2m_min", 0),
            "precipitation_probability_max_pct": value(daily, "precipitation_probability_max", 0),
            "precipitation_sum_mm": value(daily, "precipitation_sum", 0),
        },
        "model_note": "Live weather readings are separate from Monsoon Guardian's 2024-trained prediction model.",
    }
    _WEATHER_CACHE[cache_key] = (now, result)
    return result


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
