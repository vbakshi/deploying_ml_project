from sklearn.preprocessing import MinMaxScaler, PowerTransformer, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from regression_model.processing.transformers import TemporalVariableTransformer, RareLabelEncoder, TargetEncoder
from regression_model.config.core import config
import numpy as np
import pandas as pd 

numerical_pipe = Pipeline( [
    ("imputer", SimpleImputer(strategy='most_frequent')),
    ("log_transformer", FunctionTransformer(np.log))]
)

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("rare_label_encoder", RareLabelEncoder()),
    ("ordinal_encoder", TargetEncoder(config.model_config.categorical_features))
]
)

preprocessor = ColumnTransformer([("numerical_preprocessor",  numerical_pipe, config.model_config.numerical_features), 
                                  ("categorical_preprocessor", categorical_pipe, config.model_config.categorical_features),
                                  ("temporal_preprocessor", TemporalVariableTransformer("YrSold"), config.model_config.temporal_features),
                                  ("discrete", "passthrough", config.model_config.discrete_features)
                                   ])

main_pipe = Pipeline([
    ("Preprocessor", preprocessor),
    ("Scaler", MinMaxScaler()),
    ("regression", Lasso(alpha=0.005, random_state=0))
])
