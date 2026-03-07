import os
import joblib
import optuna
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_data(path):
    df = pd.read_parquet(path)

    target = "Weekly_Sales"
    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_pipeline(X_train, params):
    cat_cols = X_train.select_dtypes(include=["object", "bool"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(**params)

    return Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )


def objective(trial, X_train, X_test, y_train, y_test):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "random_state": 42,
        "n_jobs": -1,
    }

    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.set_tag("trial_number", trial.number)
        mlflow.set_tag("model_type", "RandomForestRegressor")
        mlflow.log_params(params)

        model = build_pipeline(X_train, params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)

        mlflow.log_metric("rmse", rmse)

    return rmse


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data(
        "data/processed/train_prepared.parquet"
    )

    mlflow.set_experiment("Optuna_RF_Optimization")

    with mlflow.start_run(run_name="optuna_parent"):
        study = optuna.create_study(direction="minimize")

        study.optimize(
            lambda trial: objective(trial, X_train, X_test, y_train, y_test),
            n_trials=20
        )

        best_params = study.best_params
        best_rmse = study.best_value

        mlflow.log_params(best_params)
        mlflow.log_metric("best_rmse", best_rmse)

        final_params = best_params.copy()
        final_params["random_state"] = 42
        final_params["n_jobs"] = -1

        final_model = build_pipeline(X_train, final_params)
        final_model.fit(X_train, y_train)

        final_preds = final_model.predict(X_test)
        final_rmse = mean_squared_error(y_test, final_preds, squared=False)

        mlflow.log_metric("final_rmse", final_rmse)

        os.makedirs("models", exist_ok=True)
        model_path = "models/best_model.pkl"
        joblib.dump(final_model, model_path)

        mlflow.log_artifact(model_path)
        mlflow.sklearn.log_model(final_model, artifact_path="final_model")

        print("Best params:", best_params)
        print("Best RMSE:", best_rmse)
        print("Final RMSE:", final_rmse)
        print("Saved model:", model_path)