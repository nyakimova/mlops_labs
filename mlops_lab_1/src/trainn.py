import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_pipeline(cat_cols, num_cols, n_estimators, max_depth, random_state):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def plot_feature_importance(fitted_pipe, out_path, top_n=20):
    prep = fitted_pipe.named_steps["prep"]
    model = fitted_pipe.named_steps["model"]

    try:
        feature_names = prep.get_feature_names_out()
    except Exception:
        feature_names = np.array(
            [f"f_{i}" for i in range(len(model.feature_importances_))]
        )

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names[idx][::-1], importances[idx][::-1])
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_predictions(y_true, y_pred, out_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(args):
    df = pd.read_parquet(args.data_path)

    target = "Weekly_Sales"
    X = df.drop(columns=[target])
    y = df[target]

    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    pipe = build_pipeline(
        cat_cols=cat_cols,
        num_cols=num_cols,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )

    project_dir = Path(__file__).resolve().parents[1]
    artifacts_dir = project_dir / "artifacts"
    mlflow_dir = project_dir / "mlflow_data"

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    mlflow_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"file://{mlflow_dir.resolve()}")
    mlflow.set_experiment("Walmart_Sales_MLflow_Airflow")

    with mlflow.start_run():
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("model_type", "RandomForestRegressor")

        pipe.fit(X_train, y_train)

        pred_train = pipe.predict(X_train)
        pred_test = pipe.predict(X_test)

        metrics = {
            "rmse_train": float(mean_squared_error(y_train, pred_train, squared=False)),
            "rmse_test": float(mean_squared_error(y_test, pred_test, squared=False)),
            "mae_train": float(mean_absolute_error(y_train, pred_train)),
            "mae_test": float(mean_absolute_error(y_test, pred_test)),
            "r2_train": float(r2_score(y_train, pred_train)),
            "r2_test": float(r2_score(y_test, pred_test)),
        }

        mlflow.log_metrics(metrics)

        model_path = project_dir / "model.pkl"
        metrics_path = project_dir / "metrics.json"
        fi_path = artifacts_dir / "feature_importance.png"
        cm_path = project_dir / "confusion_matrix.png"

        joblib.dump(pipe, model_path)

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        plot_feature_importance(pipe, fi_path, top_n=20)
        plot_predictions(y_test, pred_test, cm_path)

        print(
            "RMSE train:", metrics["rmse_train"], "| RMSE test:", metrics["rmse_test"]
        )
        print("MAE train:", metrics["mae_train"], "| MAE test:", metrics["mae_test"])
        print("R2 train:", metrics["r2_train"], "| R2 test:", metrics["r2_test"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    main(args)
