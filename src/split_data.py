# split the raw data
# save it in data/processed folder

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer, OneHotEncoder
from get_data import read_params

def view(data):
    data_info = pd.DataFrame({"missing":data.isna().sum(),
    "missing(%)": np.round(data.isna().sum()/data.shape[0]*100, 2),
    "unique":data.nunique(),
    "avail": data.notna().sum()
    })
    return data_info     


def transform(df, target):

    # remove target column from preproccessing
    y = df[target]
    df = df.drop([target], axis=1)

    # view categorigal columns with large number of missing value
    categ_info = view(df[df.columns[df.dtypes == 'object']])

    #drop unwanted columns
    categ_drop = categ_info[categ_info["missing(%)"] >= 45].index
    df = df.drop(list(categ_drop), axis=1)

    #fill missing value in others categorigal columns 
    df[df.columns[df.dtypes == 'object']] = df[df.columns[df.dtypes == 'object']].fillna("ffill", axis=1)

    # prepare pipline for numerical columns
    num_pipline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])

    # final preprocessing for full data
    full_pipline = ColumnTransformer([
        ("fill", num_pipline, df.columns[df.dtypes != 'object']),
        ("encode", OneHotEncoder(sparse_output=False), df.columns[df.dtypes == 'object'])
    ])

    
    data = full_pipline.fit_transform(df)
    final_data = pd.DataFrame(data)
    #print(type(final_data))
    final_data[target] = y
    return final_data


selected_feat = ['OverallQual', 'BsmtFinSF1', 'TotalBsmtSF', 'GrLivArea', 
                 'MSZoning', 'LotShape', 'LotConfig', 'Neighborhood', 'BldgType', 
                 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 
                 'ExterQual', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'KitchenQual', 
                 'GarageType', 'GarageFinish', 'GarageCond', 'SaleType', "SalePrice"]

missing = {"BsmtQual":"No Basement","BsmtExposure":"No Basement", "BsmtFinType1":"No Basement",
 "GarageType": "No Garage", "GarageFinish": "No Garage", "GarageCond": "No Garage"}

def split_and_saved_data(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    split_ratio = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]
    target = config['base']['target_col']

    row_df = pd.read_csv(raw_data_path, sep=",")

    df = row_df[selected_feat]

    #fill missing value in others categorigal columns 
    df = df.fillna(value=missing)


    #df = transform(row_df, target) will not use it for now
    train, test = train_test_split(df, test_size=split_ratio, random_state=random_state)
    train.to_csv(train_data_path, sep=",", index=False)
    test.to_csv(test_data_path, sep=",", index=False)


if __name__=="__main__":
    args = argparse.ArgumentParser()
    default_config_path = os.path.join("params.yaml")
    args.add_argument("--config", default=default_config_path)
    #args.add_argument("--datasource", default=None)

    parsed_args = args.parse_args()
    split_and_saved_data(config_path=parsed_args.config)
     