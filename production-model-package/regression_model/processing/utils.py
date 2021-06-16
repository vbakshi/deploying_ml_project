import pandas as pd 
import numpy as np 
import math 
import typing as t 
from sklearn.pipeline import Pipeline 
from sklearn.metrics import mean_squared_error , r2_score
from regression_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config
from regression_model import __version__ as _version

import joblib

def load_data(*, data_file_name: str) -> pd.DataFrame:
    return pd.read_csv(f"{DATASET_DIR}/{data_file_name}", index_col=0)

def save_pipeline(*, pipeline_to_save: Pipeline) -> None:

    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name 
    remove_old_pipelines(files_to_keep = [save_file_name])
    joblib.dump(pipeline_to_save, save_path)

def load_pipeline(*, file_name: str) -> Pipeline:

    file_path = TRAINED_MODEL_DIR / file_name 
    trained_pipeline = joblib.load(file_path)
    return trained_pipeline 

def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:

    do_not_delete = files_to_keep + ['__init__.py']
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file not in do_not_delete:
            model_file.unlink()

def score(true_labels, predictions):
    mse = mean_squared_error(np.exp(true_labels), np.exp(predictions))
    rmse = math.sqrt(mse)
    r2 = r2_score(np.exp(true_labels), np.exp(predictions))

    return mse, rmse, r2
    