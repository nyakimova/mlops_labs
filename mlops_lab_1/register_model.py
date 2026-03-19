from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

MODEL_NAME = "walmart_sales_rf"
EXPERIMENT_NAME = "Walmart_Sales_MLflow_Airflow"


def main():
    repo_dir = Path("/opt/airflow/repo")
    model_path = repo_dir / "model.pkl"
    mlruns_dir = Path("/mlflow_data")

    if not model_path.exists():
        raise FileNotFoundError(f"Не знайдено model.pkl: {model_path}")

    mlruns_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(mlruns_dir.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    model = joblib.load(model_path)

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, artifact_path="model")
        run_id = run.info.run_id

    client = MlflowClient()

    try:
        client.create_registered_model(MODEL_NAME)
    except Exception:
        pass

    model_uri = f"runs:/{run_id}/model"
    registered_model = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    try:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=registered_model.version,
            stage="Staging",
        )
        print(
            f"Model registered: name={MODEL_NAME}, "
            f"version={registered_model.version}, stage=Staging"
        )
    except Exception as e:
        print(
            f"Model registered: name={MODEL_NAME}, "
            f"version={registered_model.version}, "
            f"але перевести в Staging не вдалося: {e}"
        )


if __name__ == "__main__":
    main()