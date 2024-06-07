import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer, OneHotEncoder
from get_data import read_params
import argparse
import joblib
import json
import mlflow
from urllib.parse import urlparse

def model_pipline(df, model):

    categorical_columns = df.columns[df.dtypes == 'object']
    numerical_columns = df.columns[df.dtypes != 'object']

    model_pipline = Pipeline([
        ("preprocessor", ColumnTransformer(
            transformers=[
                ("categorical", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_columns),
                ("numerical",SimpleImputer(strategy="median"), numerical_columns)]
        )),
        ("regressor", model)
    ])

    return model_pipline


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    max_depth = config["estimators"]["RandomForestRegressor"]["params"]["max_depth"]
    max_features = config["estimators"]["RandomForestRegressor"]["params"]["max_features"]
    min_samples_split = config["estimators"]["RandomForestRegressor"]["params"]["min_samples_split"]
    n_estimators = config["estimators"]["RandomForestRegressor"]["params"]["n_estimators"]

    target = config["base"]["target_col"]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    # mlflow setup
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)

    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:

        lr = RandomForestRegressor(max_depth=max_depth, max_features=max_features, min_samples_split=min_samples_split, n_estimators=n_estimators, random_state=random_state)

        final_model = model_pipline(train_x, lr)
            
        final_model.fit(train_x, train_y)
        predicted_qualities = final_model.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        params = {
                "max_depth": max_depth,
                "max_features": max_features,
                "min_samples_split": min_samples_split,
                "n_estimators":n_estimators}
        
        mlflow.log_params(params)

        scores = {
                "rmse": rmse,
                "mae": mae,
                "r2": r2}

        mlflow.log_metrics(scores)

        tracking_url_store = urlparse(mlflow.get_artifact_uri()).scheme
        if tracking_url_store != "file":
            mlflow.sklearn.log_model(lr, "model", registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.load_model(lr, "model")


    # model_path = os.path.join(model_dir, "model.joblib")
    # joblib.dump(final_model, model_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    default_config_path = os.path.join("params.yaml")
    args.add_argument("--config", default=default_config_path)
    #args.add_argument("--datasource", default=None)
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
     