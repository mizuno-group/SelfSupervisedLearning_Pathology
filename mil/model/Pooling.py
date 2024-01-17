""" 
Max Plooling and Logistic Regression for MIL
"""

import random

import numpy as np
from sklearn.linear_model import LogisticRegression

class PoolingMIL:
    def __init__(self, n_features:int=512, strategy="max"):
        self.n_features=n_features
        self.train_data=np.zeros((0,self.n_features), dtype=np.float32)
        self.test_data=np.zeros((0,self.n_features), dtype=np.float32)
        dict_strategy={
            "max":self._max_pooling,
            "min":self._min_pooling,
            "mean":self._mean_pooling,
        }
        self.pooling=dict_strategy[strategy]
        self.logistic_model=None

    def prepare_train_data(self, X):
        self.train_data = self.pooling(self.train_data, X)    

    def prepare_test_data(self, X):
        self.test_data = self.pooling(self.test_data, X)

    def train(self, x_train=None, y_train=None, params=dict(), convertz=False,):
        self.logistic_model=LogisticRegression(**params)
        if x_train:
            self.logistic_model.fit(x_train, y_train)
        else:
            self.logistic_model.fit(self.train_data, y_train)

    def predict(self, x_test=None):
        if x_test:
            return self.logistic_model.predict_proba(x_test)[:,1]
        else:
            return self.logistic_model.predict_proba(self.test_data)[:,1]

    def save_model(self, model_path=""):
        pd.to_pickle(np.concatenate([
            self.logistic_model.coef_,
            self.logistic_model.intercept_,]),
            model_path)

    def _max_pooling(self, data, X):
        return np.concatenate([data, np.max(X, axis=1)])

    def _min_pooling(self, data, X):
        return np.concatenate([data, np.max(X, axis=1)])

    def _mean_pooling(self, data, X):
        return np.concatenate([data, np.max(X, axis=1)])

    def _delete_data(self):
        self.train_data=np.zeros((0,self.n_features), dtype=np.float32)
        self.test_data=np.zeros((0,self.n_features), dtype=np.float32)
