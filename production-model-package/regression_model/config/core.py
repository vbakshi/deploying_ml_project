from pathlib import Path 
from typing import Dict, List, Sequence 

from pydantic import BaseModel 
from strictyaml import YAML, load 

import regression_model 

PACKAGE_ROOT = Path(regression_model.__file__).resolve().parent 
ROOT = PACKAGE_ROOT.parent 
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "data"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """
    Application level config

    """
    package_name:str
    training_data_file: str 
    test_data_file: str 


class ModelConfig(BaseModel):
    """
    Configuration for model training and feature engineering
    """

    target: str 
    features: List[str]
    test_size: float
    random_state: int
    categorical_features: List[str]
    numerical_features: List[str]
    discrete_features: List[str]
    temporal_features: List[str]
    categorical_features_with_missing: List[str]
    numerical_features_with_missing: List[str]


class Config(BaseModel):
    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    """ Get config file path"""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH 
    raise Exception(f"Config not found at {CONFIG_FILE_PATH}")

def get_config_from_yaml(cfg_path: Path = None) -> YAML:

    """ Parse config.yml with package configuration """

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path) as config_file:
            parsed_config = load(config_file.read())
            return parsed_config 
    raise OSError(f"Could not find config file at path: {cfg_path}")

def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """ Run validation on config values"""

    if parsed_config is None:
        parsed_config = get_config_from_yaml()
    
    
    _config = Config(
            app_config = AppConfig(**parsed_config.data),
            model_config = ModelConfig(**parsed_config.data)
        )
    return _config

config = create_and_validate_config()