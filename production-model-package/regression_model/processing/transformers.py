import numpy as np 
import pandas as pd 
from sklearn.base import TransformerMixin, BaseEstimator


class TemporalVariableTransformer(TransformerMixin, BaseEstimator):

    def __init__ (self, reference_variable):
        # self.temp_variables = variables 
        self.reference_variable = reference_variable 

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        for var in X.columns:
            X[var] = X[self.reference_variable] - X[var]
        # for var in self.temp_variables:
        #     X[var] = X[self.reference_variable] - X[var]
        X = X.drop(self.reference_variable, axis=1)

        return X


class RareLabelEncoder(TransformerMixin, BaseEstimator):

    def __init__ (self, tol=0.01):
        # self.variables = variables 
        self.tol = tol 
        self.mapping = dict()

    
    def fit(self, X, y=None):

        for i in range(X.shape[1]):

            freq = pd.Series(X[:,i]).value_counts(normalize=True)
            self.mapping[i] = freq[freq>self.tol].index
        return self
    
    def transform(self, X):

        X = X.copy()

        for i in range(X.shape[1]):
            X[:,i] = np.where(pd.Series(X[:,i]).isin(self.mapping[i]), X[:,i], "Rare")
        
        return X

class TargetEncoder(TransformerMixin, BaseEstimator):

    def __init__(self, cols):
        self.cols = cols
        self.ordinal_label_dict = {}
        self.target_label = "target"

    def fit(self, X, y=None):
        X=pd.DataFrame(X, columns=self.cols)
        X[self.target_label] = y

        for var in self.cols:
            ordinal_labels = X.groupby(var)[self.target_label].mean().sort_values().index
            ordinal_label = {k:i for i,k in enumerate(ordinal_labels)}
            self.ordinal_label_dict[var] = ordinal_label
        return self
            
    def transform(self, X):
        X = pd.DataFrame(X, columns=self.cols)
        for var in X.columns:
            X[var] = X[var].map(self.ordinal_label_dict[var])

        return X

class Mapper(TransformerMixin, BaseEstimator):

    def __init__(self, variables, mappings):
        self.variables = variables 
        self.mappings = mappings

    def fit(self, X, y=None):
        return self 

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].map(self.mappings[var])
        return X
