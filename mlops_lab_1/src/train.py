import argparse
import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import mlflow


def load_walmart_train(train_csv):
    df = pd.read_csv(train_csv)

    required = {"Store", "Dept", "Date", "IsHoliday", "Weekly_Sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Missing columns: {}. Columns: {}".format(missing, list(df.columns))
        )

    df["Date"] = pd.to_datetime(df["Date"])
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["week"] = df["Date"].dt.isocalendar().week.astype(int)
    df = df.drop(columns=["Date"])

    df["IsHoliday"] = df["IsHoliday"].astype(str)
    return df


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
            ["f_{}".format(i) for i in range(len(model.feature_importances_))]
        )

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names[idx][::-1], importances[idx][::-1])
    plt.title("Top {} Feature Importances".format(top_n))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(args):
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / args.data_path
    artifacts_dir = base_dir / "artifacts"
    metrics_path = base_dir / "metrics.json"
    model_path = base_dir / "model.pkl"
    mlruns_dir = Path("/tmp/mlruns_tracking")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(mlruns_dir.resolve().as_uri())
    mlflow.set_experiment(args.experiment_name)

    df = load_walmart_train(data_path)

    target = "Weekly_Sales"
    X = df.drop(columns=[target])
    y = df[target]

    cat_cols = [c for c in ["Store", "Dept", "IsHoliday"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    max_depth = None if args.max_depth == -1 else args.max_depth

    pipe = build_pipeline(
        cat_cols=cat_cols,
        num_cols=num_cols,
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        random_state=args.random_state,
    )

    with mlflow.start_run():
        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)

        mlflow.set_tag("author", args.author)
        mlflow.set_tag(
            "dataset", "Walmart Recruiting - Store Sales Forecasting (train.csv)"
        )
        mlflow.set_tag("task", "regression")
        mlflow.set_tag("model_type", "RandomForest")

        pipe.fit(X_train, y_train)

        pred_train = pipe.predict(X_train)
        pred_test = pipe.predict(X_test)

        rmse_train = mean_squared_error(y_train, pred_train, squared=False)
        rmse_test = mean_squared_error(y_test, pred_test, squared=False)
        mae_train = mean_absolute_error(y_train, pred_train)
        mae_test = mean_absolute_error(y_test, pred_test)
        r2_train = r2_score(y_train, pred_train)
        r2_test = r2_score(y_test, pred_test)

        mlflow.log_metric("rmse_train", rmse_train)
        mlflow.log_metric("rmse_test", rmse_test)
        mlflow.log_metric("mae_train", mae_train)
        mlflow.log_metric("mae_test", mae_test)
        mlflow.log_metric("r2_train", r2_train)
        mlflow.log_metric("r2_test", r2_test)

        fi_path = artifacts_dir / "feature_importance.png"
        plot_feature_importance(pipe, fi_path, top_n=20)

        metrics = {
            "rmse_train": float(rmse_train),
            "rmse_test": float(rmse_test),
            "mae_train": float(mae_train),
            "mae_test": float(mae_test),
            "r2_train": float(r2_train),
            "r2_test": float(r2_test),
        }

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        joblib.dump(pipe, model_path)

        print("RMSE train:", rmse_train, "| RMSE test:", rmse_test)
        print("MAE  train:", mae_train, "| MAE  test:", mae_test)
        print("R2   train:", r2_train, "| R2   test:", r2_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw/walmart-recruiting-store-sales-forecasting/train.csv/train.csv",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="Walmart_Sales_MLflow_Airflow",
    )
    parser.add_argument("--author", type=str, default="nyakimova")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    main(args)
