import json
import os

import pandas as pd


def test_data_validation():
    data_path = os.getenv("DATA_PATH", "data/ci/train_prepared_sample.parquet")
    assert os.path.exists(data_path), f"File not found: {data_path}"

    df = pd.read_parquet(data_path)

    required_columns = [
        "Store",
        "Dept",
        "Weekly_Sales",
        "IsHoliday",
        "year",
        "month",
        "week",
    ]

    for col in required_columns:
        assert col in df.columns, f"Missing column: {col}"

    assert df.shape[0] > 0, "Dataset is empty"
    assert df["Weekly_Sales"].notna().all(), "Weekly_Sales contains missing values"


def test_artifacts_exist():
    assert os.path.exists("models/model.joblib") or os.path.exists(
        "models/best_model.pkl"
    ), "Model artifact not found"
    assert os.path.exists("artifacts/metrics.json"), "metrics.json not found"
    assert os.path.exists(
        "artifacts/feature_importance.png"
    ), "feature_importance.png not found"


def test_quality_gate():
    metrics_path = "artifacts/metrics.json"
    assert os.path.exists(metrics_path), "metrics.json not found"

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "rmse_test" in metrics, "rmse_test not found in metrics.json"

    rmse_threshold = 20000
    assert metrics["rmse_test"] <= rmse_threshold, (
        f"Quality Gate failed: rmse_test={metrics['rmse_test']} > {rmse_threshold}"
    )
