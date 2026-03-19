from datetime import datetime
import json
import os

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator

PROJECT_DIR = "/opt/airflow/repo/mlops_lab_1"
DVC_REPO_DIR = "/opt/airflow/repo"
METRICS_PATH = os.path.join(DVC_REPO_DIR, "metrics.json")
QUALITY_THRESHOLD = 0.5


def evaluate_model():
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(f"Файл metrics.json не знайдено: {METRICS_PATH}")

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    r2_value = (
        metrics.get("r2")
        or metrics.get("r2_test")
        or metrics.get("R2")
        or metrics.get("R2_test")
    )

    if r2_value is None:
        raise ValueError("У metrics.json не знайдено метрику r2 або r2_test")

    if r2_value >= QUALITY_THRESHOLD:
        return "register_model"
    return "stop_pipeline"


default_args = {
    "owner": "anastasiia",
    "depends_on_past": False,
}

with DAG(
    dag_id="ml_training_pipeline",
    default_args=default_args,
    description="ML pipeline with Airflow, DVC and MLflow",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["lab5", "mlops", "walmart"],
) as dag:

    check_data = BashOperator(
        task_id="check_data",
        bash_command=f"test -d {PROJECT_DIR}/data/raw/walmart-recruiting-store-sales-forecasting",
    )

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=f"cd {DVC_REPO_DIR} && dvc repro prepare",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"cd {DVC_REPO_DIR} && dvc repro --no-run-cache train",
    )

    evaluate_and_branch = BranchPythonOperator(
        task_id="evaluate_and_branch",
        python_callable=evaluate_model,
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command=f"cd {DVC_REPO_DIR} && python mlops_lab_1/register_model.py",
    )

    stop_pipeline = EmptyOperator(
        task_id="stop_pipeline",
    )

    check_data >> prepare_data >> train_model >> evaluate_and_branch
    evaluate_and_branch >> register_model
    evaluate_and_branch >> stop_pipeline