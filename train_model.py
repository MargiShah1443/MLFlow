import logging
import os
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


def load_diabetes_data() -> pd.DataFrame:
    dataset = load_diabetes(as_frame=True)
    features = dataset.data
    target = dataset.target
    df = features.copy()
    df["target"] = target
    return df


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    mlflow.set_experiment("mlops-diabetes-elasticnet-tuning")

    # Common split settings
    test_size = float(os.environ.get("TEST_SIZE", "0.2"))
    random_state = int(os.environ.get("RANDOM_STATE", "42"))

    data = load_diabetes_data()
    X = data.drop(columns=["target"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # ---- Hyperparameter grid ----
    alphas = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    l1_ratios = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]

    best_r2 = -1e9
    best_rmse = None
    best_mae = None
    best_alpha = None
    best_l1 = None
    best_run_id = None

    for alpha in alphas:
        for l1_ratio in l1_ratios:
            with mlflow.start_run(run_name=f"a{alpha}_l1{l1_ratio}") as run:
                logger.info(
                    "Training ElasticNet (alpha=%.4f, l1_ratio=%.2f)", alpha, l1_ratio
                )

                pipeline = Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            ElasticNet(
                                alpha=alpha,
                                l1_ratio=l1_ratio,
                                random_state=random_state,
                            ),
                        ),
                    ]
                )

                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                rmse, mae, r2 = eval_metrics(y_test, y_pred)

                print(
                    f"[Run {run.info.run_id}] alpha={alpha}, l1_ratio={l1_ratio} "
                    f"-> RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}"
                )

                # Log params & metrics
                mlflow.log_param("alpha", alpha)
                mlflow.log_param("l1_ratio", l1_ratio)
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("random_state", random_state)
                mlflow.log_param("model_type", "ElasticNet+StandardScaler")

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                # Log model artifact (each run has its own model)
                train_predictions = pipeline.predict(X_train)
                signature = infer_signature(X_train, train_predictions)

                tracking_scheme = urlparse(mlflow.get_tracking_uri()).scheme

                if tracking_scheme != "file":
                    mlflow.sklearn.log_model(
                        sk_model=pipeline,
                        artifact_path="model",
                        registered_model_name="ElasticNetDiabetesModel",
                        signature=signature,
                    )
                else:
                    mlflow.sklearn.log_model(
                        sk_model=pipeline,
                        artifact_path="model",
                        signature=signature,
                    )

                # Track best run by R2
                if r2 > best_r2:
                    best_r2 = r2
                    best_rmse = rmse
                    best_mae = mae
                    best_alpha = alpha
                    best_l1 = l1_ratio
                    best_run_id = run.info.run_id

    print("\n=== BEST MODEL (on test set) ===")
    print(f"Run ID:   {best_run_id}")
    print(f"alpha:    {best_alpha}")
    print(f"l1_ratio: {best_l1}")
    print(f"RMSE:     {best_rmse:.4f}")
    print(f"MAE:      {best_mae:.4f}")
    print(f"R2:       {best_r2:.4f}")
