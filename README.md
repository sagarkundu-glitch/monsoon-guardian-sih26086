# Monsoon Guardian — SIH26086

Hyperlocal Monsoon Onset & Break Prediction System prototype.

## Run locally

From this project folder, activate the existing Python environment and run:

```powershell
uvicorn backend.main:app --reload
```

Open <http://127.0.0.1:8000> for the dashboard. The API documentation is at
<http://127.0.0.1:8000/docs>.

## Deploy to Render

The repository includes `render.yaml`, which configures one web service for
both the dashboard and prediction API. The service needs these inference
assets in the repository:

- `data/processed/imd_rainfall_climate_multihorizon_2024.csv`
- The six `.joblib` files under `data/models/final_candidates/`

The `.gitignore` keeps these required files while excluding training data,
the virtual environment, and unrelated model artifacts. Push the project to a
GitHub repository, then in Render choose **New → Blueprint** and connect that
repository. Render reads `render.yaml`, deploys the service, and provides its
public `onrender.com` address.

The dashboard and API use the same host. Prediction models are loaded one at a
time to reduce memory use. Results remain experimental and use 2024 data; they
are not official IMD declarations.
