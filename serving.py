"""
FastAPI app for serving the ElasticNet+StandardScaler diabetes model logged with MLflow.

Usage:

1. Run training & tuning to log models:

   python train_model.py

2. From MLflow UI, pick the best run and construct a MODEL_URI like:

   runs:/<RUN_ID>/model

3. Set the environment variable:

   # macOS / Linux
   export MODEL_URI="runs:/<RUN_ID>/model"

   # Windows PowerShell
   $env:MODEL_URI="runs:/<RUN_ID>/model"

4. Start the API:

   uvicorn serving:app --reload --port 8000

5. Open:
   - Health check: http://127.0.0.1:8000/
   - Swagger UI:  http://127.0.0.1:8000/docs
"""

import os

import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------
# Load model URI from environment
# ---------------------------------------------------------------------

MODEL_URI = os.environ.get("MODEL_URI")

if not MODEL_URI:
    raise ValueError(
        "MODEL_URI environment variable is not set. "
        "Please export MODEL_URI='runs:/<RUN_ID>/model' before starting the server."
    )

# Load the model as a generic MLflow pyfunc model
model = mlflow.pyfunc.load_model(MODEL_URI)

# ---------------------------------------------------------------------
# FastAPI app & request/response schemas
# ---------------------------------------------------------------------

app = FastAPI(
    title="Diabetes Regression MLflow Serving",
    description=(
        "Serves an ElasticNet regression model trained on the "
        "sklearn diabetes dataset and logged with MLflow."
    ),
    version="1.0.0",
)


class DiabetesFeatures(BaseModel):
    # Feature names match sklearn's load_diabetes(as_frame=True).data columns
    age: float = Field(..., description="Normalized age")
    sex: float = Field(..., description="Normalized sex indicator")
    bmi: float = Field(..., description="Body mass index")
    bp: float = Field(..., description="Average blood pressure")
    s1: float = Field(..., description="TC, T-Cells or related lab feature")
    s2: float = Field(..., description="LDL or related lab feature")
    s3: float = Field(..., description="HDL or related lab feature")
    s4: float = Field(..., description="TCH or related lab feature")
    s5: float = Field(..., description="LTG or related lab feature")
    s6: float = Field(..., description="GLU or related lab feature")


class PredictionResponse(BaseModel):
    prediction: float
    model_uri: str


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------


@app.get("/")
def healthcheck():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Diabetes regression model is ready.",
        "model_uri": MODEL_URI,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: DiabetesFeatures):
    """Predict target value for given diabetes features."""
    # Convert input to a DataFrame with a single row
    df = pd.DataFrame([features.dict()])
    y_pred = model.predict(df)[0]

    return PredictionResponse(
        prediction=float(y_pred),
        model_uri=MODEL_URI,
    )
