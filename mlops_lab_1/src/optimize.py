import optuna
import pandas as pd
import numpy as np

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

    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )
    return pipe


def objective(trial, X_train, X_test, y_train, y_test):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "random_state": 42,
        "n_jobs": -1,
    }

    model = build_pipeline(X_train, params)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, preds, squared=False)

    return rmse


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data(
        "data/processed/train_prepared.parquet"
    )

    study = optuna.create_study(direction="minimize")

    study.optimize(
        lambda trial: objective(trial, X_train, X_test, y_train, y_test),
        n_trials=20
    )

    print("Best params:", study.best_params)
    print("Best RMSE:", study.best_value)
    print("Number of finished trials:", len(study.trials))