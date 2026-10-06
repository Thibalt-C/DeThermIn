import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold

def format_for_learning(dataset, features, no_y=False):

    binary_features = ["mineralized", "vegetalized", "artificial", "buildings_mask", "water_mask", "cloud_mask", "shadow"]

    X_continuous = []
    X_binary = []

    for feature in features:
        x = dataset[feature].values.reshape(-1, 1)
        if np.isin(feature, binary_features):
            x = x.astype(int) # binary 0 and 1
            #x[np.isnan(x)] = 0  # Handle NaN for binary features
            X_binary.append(x)
        else:
            X_continuous.append(x)

    X_continuous = np.concatenate(X_continuous, axis=1)

    # Concatenate back
    if len(X_binary)!=0:
        X_binary = np.concatenate(X_binary, axis=1)
        X = np.concatenate([X_binary, X_continuous], axis=1)
    else:
        X = X_continuous

    if not no_y:
        y = dataset.LST_ground.values.reshape(-1, 1)
        nan_mask = (np.isnan(X).any(axis=1)) | (np.isnan(y).squeeze())
        y = y[~nan_mask]
        X = X[~nan_mask]
        return X, y, nan_mask

    else:
        nan_mask = np.isnan(X).any(axis=1)
        X = X[~nan_mask]
        return X, nan_mask

def format_for_deep_learning(dataset, features, scaler=None, no_y=False):

    binary_features = ["mineralized", "vegetalized", "artificial", "buildings_mask", "water_mask", "cloud_mask", "shadow"]

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


def reconstruct(y_pred_test, nan_mask, train_size, dataset, coarse_LST_layer = "fine_LST"):

    finite_indices = np.where(~nan_mask)[0]
    test_indices = finite_indices[int(len(finite_indices) * train_size):]
    test = np.full(len(nan_mask), np.nan)
    test[test_indices] = y_pred_test
    test = test.reshape(len(dataset.y), len(dataset.x))

    LST_coarse = dataset[coarse_LST_layer].values.flatten()
    LST_coarse[~test_indices] = np.nan
    LST_coarse = LST_coarse.reshape(len(dataset.y), len(dataset.x))

    LST_ground = dataset.LST_ground.values.flatten()
    LST_ground[~test_indices] = np.nan
    LST_ground = LST_ground.reshape(len(dataset.y), len(dataset.x))

    LST_ground = LST_ground[~np.isnan(test).all(axis=1), :]
    LST_coarse = LST_coarse[~np.isnan(test).all(axis=1), :]
    test = test[~np.isnan(test).all(axis=1), :]

    residuals = test - LST_ground

    return LST_ground, LST_coarse, test, residuals

def reconstruct_for_DL(y_pred_test, dataset, coarse_LST_layer="fine_LST"):

    H, W = dataset.rio.shape
    h, w = y_pred_test.shape

    test = y_pred_test
    LST_coarse = dataset[coarse_LST_layer].values[H-h:,:]
    LST_ground = dataset.LST_ground.values[H-h:,:]
    residuals = test - LST_ground

    return LST_ground, LST_coarse, test, residuals


def reconstruct_full(dataset, features, model, layer_name):
    cube = []

    for time in dataset.time.values:
        sample = dataset.sel(time=time)

        X, nan_mask = format_for_learning(sample, features, no_y=True)

        reconstructed = np.full(len(nan_mask), np.nan)
        reconstructed[~nan_mask] = model.predict(X)
        reconstructed = reconstructed.reshape(len(sample.y), len(sample.x))

        cube.append(reconstructed)

    cube = np.stack(cube, axis=-1)

    dataset[layer_name] = xr.DataArray(cube, dims=("y", "x", "time"),
                                            coords={"time": dataset.time, "y": dataset.y, "x": dataset.x},
                                            attrs={"units": "°C", "standard_name": "Land Surface Temperature",
                                                   "plot_kwargs": {"cmap": "RdYlBu_r"}})

    return dataset


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
    cv = KFold(n_splits=cv_folds, shuffle=False)
    cv_scores = cross_val_score(model, X_train, y_train.squeeze(), cv=cv, scoring='neg_root_mean_squared_error')
    print(f"Cross-validated RMSE scores: {-cv_scores}")
    print(f"Mean CV RMSE: {-cv_scores.mean():.2f} °C")