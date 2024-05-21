import os 
import argparse
import yaml
import logging
import pandas as pd


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config = read_params(config_path)
    #print(config)
    data_path = config["data_source"]["s3_source"]
    df = pd.read_csv(data_path, sep=",", encoding="utf-8")
    #print(df.head())
    return df

if __name__=="__main__":
    args = argparse.ArgumentParser()
    default_config_path = os.path.join("params.yaml")
    args.add_argument("--config", default=default_config_path)
    #args.add_argument("--datasource", default=None)

    parsed_args = args.parse_args()
    get_data(config_path=parsed_args.config)