from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from .preprocessing import get_scaler, format_for_deep_learning
from .error_quantifiers import rmse
from .CNNs import extract_patches, CustomDataset, reconstruct_DL
from torch.utils.data import DataLoader
import numpy as np
import torch
# noinspection PyUnusedImports
import rioxarray


class CustomLinearRegression(BaseEstimator, RegressorMixin):
    """
    Linear model with gradient descent.
    """
    def __init__(self, learning_rate=0.1, max_iterations=1000, step_threshold=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.step_threshold = step_threshold
        self.weights = None
        self.scaler = None
        self.train_errors = []

    def fit(self, X, y, is_2D=False):

        if is_2D:
            H, W, n_features = X.shape
            nan_mask = (np.isnan(X).any(axis=2)) | np.isnan(y.squeeze())
            X = X[~nan_mask, :].reshape(-1, n_features)
            y = y[~nan_mask].reshape(-1, 1).squeeze()

        # scale
        self.scaler = get_scaler(X)
        X = self.scaler.fit_transform(X)

        # Add bias term
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        self.weights = np.random.randn(X.shape[1])
        self.train_errors = [np.inf]

        for i in range(self.max_iterations):
            y_hat = np.dot(X, self.weights)
            self.weights -= self.learning_rate * np.dot(X.T, (y_hat - y.squeeze())) / len(y)
            train_error = rmse(y.squeeze(), y_hat)
            self.train_errors.append(train_error)
            if self.train_errors[-2] - self.train_errors[-1] < self.step_threshold:
                break
        return self

    def predict(self, X, return_2D=False):

        if return_2D:
            H, W, n_features = X.shape
            nan_mask = (np.isnan(X).any(axis=2))
            X = X[~nan_mask, :].reshape(-1, n_features)


        X = self.scaler.transform(X)
        if X.shape[1] == len(self.weights) - 1:
            X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

        y_pred = np.dot(X, self.weights)

        if return_2D:
            y_reshaped = np.full((H, W), np.nan)
            y_reshaped[~nan_mask] = y_pred
            return y_reshaped
        else:
            return y_pred.squeeze()



class CustomRandomForestRegressor(BaseEstimator, RegressorMixin):
    """
    Random forest regressor.
    """

    def __init__(self, n_estimators=10, max_depth=10, max_features=None):
        self.scaler = None
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.regressor = RandomForestRegressor(n_estimators=self.n_estimators,
                                               max_depth=self.max_depth,
                                               max_features=self.max_features)

    def fit(self, X, y):
        self.scaler = get_scaler(X)
        X = self.scaler.fit_transform(X)
        self.regressor.fit(X, y)
        return self

    def predict(self, X):
        X = self.scaler.transform(X)
        y_pred = self.regressor.predict(X)
        return y_pred

    def score(self, X, y, **kwargs):
        y_pred = self.predict(X)
        return -rmse(y.squeeze(), y_pred)  # Negative for minimization



class TrainedResUNet:
    """
    U-Net based deep neural network with residual blocks.
    """
    def __init__(self, model, features, batch_size, epochs, patch_size, scaler):
        self.features = features
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.stride = self.patch_size//2
        self.epochs = epochs
        self.regressor = model
        self.scaler = scaler

    def predict(self, dataset):
        X, _ = format_for_deep_learning(dataset, self.features, scaler=self.scaler, no_y=True)
        y = np.zeros((X.shape[0], X.shape[1], 1))
        X, y, valid_patches = extract_patches(X, y, patch_size=self.patch_size, stride=self.stride)
        X = torch.tensor(X)
        y = torch.tensor(y)
        dataloader = DataLoader(CustomDataset(X,y),
                                batch_size=self.batch_size, shuffle=False,
                                num_workers=0, persistent_workers=False)
        H, W = dataset.rio.shape
        y_pred = reconstruct_DL(self.regressor, dataloader, self.patch_size, self.stride, H, W, valid_patches)

        return y_pred