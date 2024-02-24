# load the train and test
# train algo
# save metrices, params

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from get_data import read_params
import argparse
import joblib
import json


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

    lr = RandomForestRegressor(max_depth=max_depth, max_features=max_features, min_samples_split=min_samples_split, n_estimators=n_estimators, random_state=random_state)
    
    lr.fit(train_x, train_y)
    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("RandomForestRegressor model (max_depth=%f, max_features=%f, min_samples_split=%f, n_estimators=%f):" % (max_depth, max_features, min_samples_split, n_estimators))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]
    
    with open(scores_file, "w") as f:
        scores = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f:
        params = {
            "max_depth": max_depth,
            "max_features": max_features,
            "min_samples_split": min_samples_split,
            "n_estimators":n_estimators
        }
        json.dump(params, f, indent=4)

    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(lr, model_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    default_config_path = os.path.join("params.yaml")
    args.add_argument("--config", default=default_config_path)
    #args.add_argument("--datasource", default=None)
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
     