
from os import replace
import pandas as pd 
import numpy as np 
import math 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error , r2_score
from math import sqrt 
from regression_model.config.core import DATASET_DIR

import joblib


def load_data(file_name:str) -> pd.DataFrame:
    return pd.read_csv(f"{DATASET_DIR}/{file_name}", index_col=0)


def divide_train_test(data, target):
    X_train, X_test, y_train, y_test = train_test_split(data, data[target], test_size=0.1, random_state=0)
    return X_train, X_test, y_train, y_test 


def impute_na(df, var, replacement='missing'):
    return df[var].fillna(replacement)


def elapsed_years(df, var, ref_var='YrSold'):
    X = df.copy()
    X[var] = X[ref_var] - X[var]
    return X 


def log_transform(df, var):
    return np.log(df[var])


def remove_rare_labels(df, var, frequent_labels):
    return np.where(df[var].isin(frequent_labels), df[var], "Rare")


def encode_categorical(df, var, mappings):
    return df[var].map(mappings)


def train_scaler(df, output_path):
    scaler = MinMaxScaler()
    scaler.fit(df)
    joblib.dump(scaler, output_path)
    return scaler


def scale_features(df, scaler_path):
    scaler = joblib.load(scaler_path)
    return scaler.transform(df)


def train_model(df, target, output_path):

    lin_model = Lasso(alpha=0.005, random_state=0) 

    lin_model.fit(df,target)

    joblib.dump(lin_model, output_path)

    return lin_model

def predict(df, model):
    lin_model = joblib.load(model)
    return lin_model.predict(df)

def score(true_labels, predictions):
    mse = mean_squared_error(np.exp(true_labels), np.exp(predictions))
    rmse = math.sqrt(mse)
    r2 = r2_score(np.exp(true_labels), np.exp(predictions))

    return mse, rmse, r2
    