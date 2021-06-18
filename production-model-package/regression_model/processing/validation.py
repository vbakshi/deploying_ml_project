from typing import List, Optional, Tuple 

import numpy as np 
import pandas as pd 

from pydantic import BaseModel, ValidationError 

from regression_model.config.core import config 

def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:

    """Check for na values"""

    validated_data = input_data.copy() 
    new_vars_with_na = [

        var 
        for var in config.model_config.features

        if var not in config.model_config.numerical_features_with_missing +
        config.model_config.categorical_features_with_missing and 
        validated_data[var].isnull().sum()  > 0 
    ]

    validated_data.dropna(subset=new_vars_with_na, inplace=True)

    return validated_data 



def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:

    input_data.rename(columns=config.model_config.features_to_rename, inplace=True)
    relevant_data = input_data[config.model_config.features].copy() 
    validated_data = drop_na_inputs(input_data=relevant_data) 
    errors = None 

    try:
        MultipleHousePriceInputs(
            inputs = validated_data.replace({np.nan:None}).to_dict(orient='records')
        )
    except ValidationError as error:
        errors = error.json() 
    
    return validated_data, errors

class HousePriceInputSchema(BaseModel):

    MSSubClass: Optional[int]
    MSZoning: Optional[str]
    Neighborhood: Optional[str]
    OverallQual: Optional[int]
    OverallCond: Optional[int]
    YearRemodAdd: Optional[int]
    RoofStyle: Optional[str]
    MasVnrType: Optional[str]
    BsmtQual: Optional[str]
    BsmtExposure: Optional[str]
    HeatingQC: Optional[str]
    CentralAir: Optional[str]
    FirstFlrSF: Optional[int]
    GrLivArea: Optional[int]
    BsmtFullBath: Optional[float]
    KitchenQual: Optional[str]
    Fireplaces: Optional[int]
    FireplaceQu: Optional[str]
    GarageType: Optional[str]
    GarageFinish: Optional[str]
    GarageCars: Optional[float]
    PavedDrive: Optional[str]
    LotFrontage: Optional[float]
    YrSold: Optional[int]

class MultipleHousePriceInputs(BaseModel):
    inputs: List[HousePriceInputSchema]