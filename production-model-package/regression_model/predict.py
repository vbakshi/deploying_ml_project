import typing as t 

import numpy as np 
import pandas as pd 

from regression_model import __version__ as _version 
from regression_model.config.core import config 
from regression_model.processing.utils import load_pipeline, load_data, score
from regression_model.processing.validation import validate_inputs 


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_price_pipe = load_pipeline(file_name=pipeline_file_name)


def make_predictions(
    *,
    input_data: t.Union[pd.DataFrame, dict]
) -> dict:

    data = pd.DataFrame(input_data)
    print(data.dtypes)
    validated_data, errors = validate_inputs(input_data=data)
    # print(validated_data.dtypes)
    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        predictions = _price_pipe.predict(X = validated_data)
        results["predictions"] = [np.exp(prediction) for prediction in predictions]

    return results

if __name__ == '__main__':

    test_data = load_data(data_file_name='test.csv')
    results = make_predictions(input_data = test_data[config.model_config.features])
