# MLflow Lab – Diabetes Regression with ElasticNet & Model Serving

- **Diabetes regression** using the `sklearn.datasets.load_diabetes` dataset  
- **Standardized features** via `StandardScaler`  
- **ElasticNet regression** with L1/L2 regularization  
- **MLflow experiment tracking** (parameters, metrics, model artifacts)  
- **Hyperparameter tuning** over `alpha` and `l1_ratio`, logged as multiple MLflow runs  
- **Model serving** using a FastAPI app that loads an MLflow model and exposes a `/predict` endpoint

---

## 📁 Repository Structure

```text
MLFlow-Lab/
│
├── train_model.py      # Training + hyperparameter tuning + MLflow logging
├── serving.py          # FastAPI app for serving an MLflow-logged model
├── requirements.txt    # Python dependencies
└── README.md           # This file
````

---

## ⚙️ 1. Environment Setup & Install Dependencies

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The key libraries are:

* `mlflow`
* `scikit-learn`
* `numpy`
* `pandas`
* `fastapi`
* `uvicorn`

---

## 🧪 2. Train & Tune the Model with MLflow

The training script:

* Loads the **diabetes dataset**
* Splits into train / test sets
* Builds a **Pipeline**: `StandardScaler` → `ElasticNet`
* Runs a small **grid search** over `(alpha, l1_ratio)`
* Logs **each combination** as a separate MLflow run
* Logs:

  * Parameters: `alpha`, `l1_ratio`, `test_size`, `random_state`, `model_type`
  * Metrics: `rmse`, `mae`, `r2`
  * The trained model as an MLflow model artifact
* Prints the **best run** (based on highest R²)

Run:

```bash
python train_model.py
```

You will see console output like:

```text
[Run ...] alpha=0.1, l1_ratio=0.1 -> RMSE=53.48, MAE=42.95, R2=0.4600
...
=== BEST MODEL (on test set) ===
Run ID:   8345dbe459d44056aad2a8868c3d7aec
alpha:    0.5
l1_ratio: 0.7
RMSE:     53.3095
MAE:      43.0460
R2:       0.4636
```

(Your exact numbers and best run ID may differ, but should be in a similar range.)

This shows:

* A realistic R² around ~0.46 for this small/ noisy dataset
* How proper **scaling + regularization + tuning** improve performance vs. a naive baseline

---

## 📈 3. Inspect Experiments in MLflow UI

Start the MLflow UI:

```bash
mlflow ui
```

Then open in browser:

👉 `http://127.0.0.1:5000`

You should see:

* An experiment named something like:
  **`mlops-diabetes-elasticnet-tuning`**
* Multiple runs (one per `(alpha, l1_ratio)` pair)
* For each run:

  * Parameters (`alpha`, `l1_ratio`, etc.)
  * Metrics (`rmse`, `mae`, `r2`)
  * A logged model under **Artifacts → model**

For your report/screenshots, you can:

1. Sort runs by **R² descending** to highlight the best run.
2. Click the best run and show its parameters/metrics/model artifact.

---

## 🤖 4. Serve the Best Model with FastAPI

The `serving.py` script loads a model using `mlflow.pyfunc.load_model`
and exposes a **REST API** for predictions.

### 4.1 Choose a Model URI

From the MLflow UI:

1. Open the **best run**.
2. Under **Artifacts → model**, click the **model**.
3. Copy the run ID from the URL or from the run details.

Construct the model URI as:

```text
runs:/<RUN_ID>/model
```

For example:

```text
runs:/8345dbe459d44056aad2a8868c3d7aec/model
```

### 4.2 Export `MODEL_URI` and Start the Server

Set the `MODEL_URI` environment variable and run FastAPI via Uvicorn:

```bash
# macOS / Linux
export MODEL_URI="runs:/<RUN_ID>/model"

# Windows PowerShell
$env:MODEL_URI="runs:/<RUN_ID>/model"
```

Then start the server:

```bash
uvicorn serving:app --reload --port 8000
```

The API will be available at:

* Root health check: `http://127.0.0.1:8000/`
* Interactive docs (Swagger): `http://127.0.0.1:8000/docs`

---

## 🧪 5. Example Prediction Request

The diabetes dataset has the following feature columns:

* `age`, `sex`, `bmi`, `bp`, `s1`, `s2`, `s3`, `s4`, `s5`, `s6`

All are **numeric** (already scaled-like values from sklearn).

You can send a sample request using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "age": 0.02,
        "sex": -0.04,
        "bmi": 0.03,
        "bp": 0.02,
        "s1": -0.01,
        "s2": 0.0,
        "s3": -0.02,
        "s4": 0.03,
        "s5": 0.02,
        "s6": 0.0
      }'
```

You should receive a JSON response similar to:

```json
{
  "prediction": 161.23,
  "model_uri": "runs:/<RUN_ID>/model"
}
```

This `prediction` corresponds to the regression target in the diabetes dataset.

---

## 📝 Summary

This lab demonstrates:

* Using **MLflow** for experiment tracking:

  * parameters, metrics, and model artifacts
* Improving a model via:

  * **feature scaling**
  * **ElasticNet regularization**
  * **hyperparameter tuning**
* Serving an MLflow model with **FastAPI**, exposing a simple `/predict` endpoint