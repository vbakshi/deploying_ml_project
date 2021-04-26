import os
import sys
import pandas as pd
import joblib
import utils as utils
import config as config
import warnings 
# import procedure_prgming.config as config
import numpy as np
warnings.simplefilter(action='ignore')


mode_dict=np.load(config.MODE_PATH, allow_pickle=True).item()

features=config.SELECTED_FEATURES + [config.TARGET]
frequent_labels_dict=np.load(config.FREQUENT_VARS_DICT, allow_pickle=True).item()
ordinal_labels_dict=np.load(config.ORDINAL_LABEL_DICT, allow_pickle=True).item()

print("Loading Data ...")
print()
data = utils.load_data(config.FILE_PATH)

print("Splitting data into train and test ...")
print()
X_train, X_test, y_train, y_test = utils.divide_train_test(data, config.TARGET)

print("Encoding missing values ...")
print()
for var in config.CATEGORICAL_VARIABLES_NA:
    X_train[var] = utils.impute_na(X_train, var)
    X_test[var] = utils.impute_na(X_test, var)

for var in config.NUMERICAL_VARIABLES_NA:
    X_train[var] = utils.impute_na(X_train,var, mode_dict[var])
    X_test[var] = utils.impute_na(X_test, var, mode_dict[var])

print("Transforming temporal variables ...")
print()
for var in config.TEMPORAL_VARIABLES:
    print(var)
    X_train = utils.elapsed_years(X_train, var)
    X_test = utils.elapsed_years(X_test, var)

print("Transforming numerical variables using Log transform ...")
print()
for var in config.TRANSFORM_VARIABLES:
    X_train[var] = utils.log_transform(X_train, var)
    X_test[var] = utils.log_transform(X_test, var)

print("Grouping rare labels ...")
print()
for var in frequent_labels_dict.keys():
    X_train[var] = utils.remove_rare_labels(X_train, var, frequent_labels_dict[var])
    X_test[var] = utils.remove_rare_labels(X_test, var, frequent_labels_dict[var])

print("Encoding categorical variables using target encoding")
print()
for var in ordinal_labels_dict.keys():
    X_train[var] = utils.encode_categorical(X_train, var, ordinal_labels_dict[var])
    X_test[var] = utils.encode_categorical(X_test, var, ordinal_labels_dict[var])

print("Train scaler and transform data...")
scaler = utils.train_scaler(X_train[config.SELECTED_FEATURES], config.SCALER_PATH)

y_train = X_train[config.TARGET]
y_test = X_test[config.TARGET]

X_train = pd.DataFrame(utils.scale_features(X_train[config.SELECTED_FEATURES], config.SCALER_PATH), columns=config.SELECTED_FEATURES)
X_test = pd.DataFrame(utils.scale_features(X_test[config.SELECTED_FEATURES], config.SCALER_PATH), columns=config.SELECTED_FEATURES)

print("Training LASSO regression model ...")
model = utils.train_model(X_train,y_train, config.MODEL_PATH)
predictions=model.predict(X_test)

mse, rmse, r2 = utils.score(y_test, predictions)

print("TEST SCORES: ")
print("Mean Squared Error: {}".format(mse))
print("Root mean Squared Error: {}".format(rmse))
print("R-Square: {}".format(r2))

predictions=model.predict(X_train)

mse, rmse, r2 = utils.score(y_train, predictions)

print("TRAIN SCORES: ")
print("Mean Squared Error: {}".format(mse))
print("Root mean Squared Error: {}".format(rmse))
print("R-Square: {}".format(r2))




