import os
import random
import joblib
import optuna
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import hydra

from omegaconf import DictConfig, OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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


def make_sampler(name: str, seed: int):
    if name.lower() == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name.lower() == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError("sampler must be 'tpe' or 'random'")


def objective_factory(cfg: DictConfig, X_train, X_test, y_train, y_test):
    def objective(trial):
        space = cfg.hpo.random_forest

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                space.n_estimators.low,
                space.n_estimators.high,
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                space.max_depth.low,
                space.max_depth.high,
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                space.min_samples_split.low,
                space.min_samples_split.high,
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                space.min_samples_leaf.low,
                space.min_samples_leaf.high,
            ),
            "random_state": cfg.seed,
            "n_jobs": -1,
        }

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.set_tag("trial_number", trial.number)
            mlflow.set_tag("model_type", cfg.model.type)
            mlflow.set_tag("sampler", cfg.hpo.sampler)
            mlflow.set_tag("seed", cfg.seed)

            mlflow.log_params(params)

            model = build_pipeline(X_train, params)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            rmse = mean_squared_error(y_test, preds, squared=False)

            mlflow.log_metric("rmse", rmse)
            return rmse

    return objective


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    set_global_seed(cfg.seed)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    X_train, X_test, y_train, y_test = load_data(cfg.data.processed_path)

    sampler = make_sampler(cfg.hpo.sampler, cfg.seed)

    with mlflow.start_run(run_name=f"hpo_parent_{cfg.hpo.sampler}"):
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config_resolved.json")

        study = optuna.create_study(
            direction=cfg.hpo.direction,
            sampler=sampler
        )

        objective = objective_factory(cfg, X_train, X_test, y_train, y_test)
        study.optimize(objective, n_trials=cfg.hpo.n_trials)

        best_params = study.best_params
        best_rmse = study.best_value

        mlflow.log_params(best_params)
        mlflow.log_metric("best_rmse", best_rmse)

        final_params = best_params.copy()
        final_params["random_state"] = cfg.seed
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


if __name__ == "__main__":
    main()