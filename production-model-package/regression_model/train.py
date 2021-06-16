import os 
import sys 
import numpy as np
import pandas as pd 
from regression_model.processing.utils import load_data, score, save_pipeline
from regression_model.config.core import config
from regression_model.pipeline import preprocessor, numerical_pipe, main_pipe
from sklearn.model_selection import train_test_split

def run_training() -> None:


    data = load_data(data_file_name = config.app_config.training_data_file)

    X_train, X_test, y_train, y_test = train_test_split(data[config.model_config.features],
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        random_state= config.model_config.random_state
    )

    y_train = numerical_pipe.fit_transform(y_train.values.reshape(-1,1))
    y_test = numerical_pipe.fit_transform(y_test.values.reshape(-1,1))


    main_pipe.fit(X_train, y_train)

    train_predictions = main_pipe.predict(X_train)
    test_predictions = main_pipe.predict(X_test)

    mse, rmse, r2 = score(y_test, test_predictions)

    print("TEST SCORES: ")
    print("Mean Squared Error: {}".format(mse))
    print("Root mean Squared Error: {}".format(rmse))
    print("R-Square: {}".format(r2))

    mse, rmse, r2 = score(y_train, train_predictions)

    print("TRAIN SCORES: ")
    print("Mean Squared Error: {}".format(mse))
    print("Root mean Squared Error: {}".format(rmse))
    print("R-Square: {}".format(r2))

    # persist trained model
    save_pipeline(pipeline_to_save = main_pipe)


if __name__=="__main__":
    run_training()
