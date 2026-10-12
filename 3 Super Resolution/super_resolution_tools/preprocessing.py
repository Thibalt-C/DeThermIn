import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold

binary_features = ["mineralized", "vegetalized", "artificial", "buildings_mask", "water_mask", "cloud_mask", "shadow", "orthoimage_shadow"]


def format_for_learning(dataset, features, no_y=False):

    X_continuous = []
    X_binary = []

    for feature in features:
        x = dataset[feature].values
        if np.isin(feature, binary_features):
            x = x.astype(int) # binary 0 and 1
            X_binary.append(x)
        else:
            X_continuous.append(x)

    X_continuous = np.stack(X_continuous, axis=-1)

    # Concatenate back
    if len(X_binary)!=0:
        X_binary = np.stack(X_binary, axis=-1)
        X = np.concatenate([X_binary, X_continuous], axis=2)
    else:
        X = X_continuous

    if not no_y:
        y = dataset.LST_ground.values
        nan_mask = (np.isnan(X).any(axis=2)) | np.isnan(y)
        y[nan_mask] = np.nan
        y = y[:, :, np.newaxis]
        X[nan_mask, :] = np.nan
        return X, y
    else:
        nan_mask = (np.isnan(X).any(axis=2))
        X[nan_mask, :] = np.nan
        return X


def format_for_deep_learning(dataset, features, scaler=None, no_y=False):

    X_continuous = []
    X_binary = []

    means = []
    stds = []
    continuous_features = []

    for feature in features:
        x = dataset[feature].values
        if np.isin(feature, binary_features):
            x = x.astype(int) # binary 0 and 1
            X_binary.append(x)

        else:
            if scaler is None:
                continuous_features.append(feature)
                means.append(np.nanmean(x))
                stds.append(np.nanstd(x))
                x = (x - np.nanmean(x)) / np.nanstd(x)
            else:
                x = (x-scaler[feature][0]) / scaler[feature][1]

            X_continuous.append(x)

    if scaler is None:
        scaler = {feature : [means[i], stds[i]] for i, feature in enumerate(continuous_features)}

    X_continuous = np.stack(X_continuous, axis=-1)

    # Concatenate back
    if len(X_binary)!=0:
        X_binary = np.stack(X_binary, axis=-1)
        X = np.concatenate([X_binary, X_continuous], axis=2)
    else:
        X = X_continuous

    if not no_y:
        y = dataset.LST_ground.values
        nan_mask = (np.isnan(X).any(axis=2)) | np.isnan(y)
        y[nan_mask] = np.nan
        y = y[:, :, np.newaxis]
        X[nan_mask, :] = np.nan
        return X, y, scaler
    else:
        nan_mask = (np.isnan(X).any(axis=2))
        X[nan_mask, :] = np.nan
        return X, scaler


def get_scaler(X):

    uniques = np.array([len(np.unique(X[:, i])) for i in range(X.shape[1])])
    binary = np.where(uniques <=2)[0]
    continuous = np.where(uniques > 2)[0]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), continuous),
            ('bin', MinMaxScaler(), binary)
        ])

    return preprocessor


def train_test_split(X, y, dataset, valid=True):

    test_mask = ~dataset.test_mask.values
    X_test = X.copy(); y_test = y.copy()
    X_test[test_mask, :] = np.nan
    y_test[test_mask] = np.nan

    if valid:
        train_mask = ~dataset.train_mask.values
        valid_mask = ~dataset.valid_mask.values
        X_train = X.copy(); X_train[train_mask, :] = np.nan
        y_train = y.copy(); y_train[train_mask] = np.nan
        X_valid = X.copy(); X_valid[valid_mask, :] = np.nan
        y_valid = y.copy(); y_valid[valid_mask] = np.nan
        return X_train, X_test, X_valid, y_train, y_test, y_valid
    else:
        train_mask = (~dataset.train_mask.values) & (~dataset.valid_mask.values)
        X_train = X.copy(); X_train[train_mask, :] = np.nan
        y_train = y.copy(); y_train[train_mask] = np.nan
        return X_train, X_test, y_train, y_test


def do_grid_search(model, param_grid, X_train, y_train, cv_folds=5):

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring='neg_root_mean_squared_error',
        verbose=2
    )
    grid_search.fit(X_train, y_train.squeeze())
    model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV RMSE: {-grid_search.best_score_:.2f} °C")

    return model


def cross_val(model, X_train, y_train, cv_folds=5):
    nan_mask = np.isnan(X_train).any(axis=2)
    X_train = X_train[~nan_mask, :].reshape(-1, X_train.shape[2])
    y_train = y_train[~nan_mask].reshape(-1, 1).squeeze()
    cv = KFold(n_splits=cv_folds, shuffle=False)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
    print(f"Cross-validated RMSE scores: {-cv_scores}")
    print(f"Mean CV RMSE: {-cv_scores.mean():.2f} °C")