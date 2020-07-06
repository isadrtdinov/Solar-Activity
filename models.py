import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from transformers import VectorTransformer
from copy import deepcopy


class LinearModel:
    def __init__(self, scale=True, x_transform=None, y_transform=None, y_inverse_transform=None):
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.y_inverse_transform = y_inverse_transform
        self.scaler = StandardScaler() if scale else None
        self.regressor = LinearRegression(fit_intercept=False)
    
    def fit(self, X, y):
        X_trans = self.x_transform(X) if self.x_transform else X
        X_trans = self.scaler.fit_transform(X_trans) if self.scaler else X_trans
        X_trans = np.append(X_trans, np.ones(X_trans.shape[0]).reshape((-1, 1)), axis=1)
        
        y_trans = self.y_transform(y) if self.y_transform else y
        self.regressor.fit(X_trans, y_trans)
        return self
    
    def predict(self, X):
        X_trans = self.x_transform(X) if self.x_transform else X
        X_trans = self.scaler.transform(X_trans) if self.scaler else X_trans
        X_trans = np.append(X_trans, np.ones(X_trans.shape[0]).reshape((-1, 1)), axis=1)
        
        y_pred_trans = self.regressor.predict(X_trans)
        y_pred = self.y_inverse_transform(y_pred_trans) if self.y_inverse_transform else y_pred_trans
        return y_pred


class MultiDimensionalRegressor:
    def __init__(self, dimensions=np.arange(1, 6), fit_decrease=False, regressor=KNeighborsRegressor, **kwargs):
        self.dimensions = dimensions
        self.regressor = regressor(**kwargs)
        self.fit_decrease = fit_decrease
    
    def fit(self, histories):
        self.regressors = []
        for dim in self.dimensions:
            vec_trans = VectorTransformer(length=dim)
            X, y = vec_trans.transform(histories)
            if self.fit_decrease:
                mask = np.all(np.diff(np.concatenate((X, y.reshape(-1, 1)), axis=1)) < 0, axis=-1)
                X, y = X[mask], y[mask]
            regr = deepcopy(self.regressor)
            regr.fit(X, y)
            self.regressors.append(regr)
        return self
    
    def predict(self, X):
        y = np.zeros(X.shape[0])
        counts = (X != 0).sum(axis=1)
        for (dim, regr) in zip(self.dimensions, self.regressors):
            mask = (counts == dim)
            if mask.sum() == 0:
                continue
            y[mask] = regr.predict(X[mask, -dim:])
        return y