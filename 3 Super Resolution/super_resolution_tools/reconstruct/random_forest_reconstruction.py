from ..preprocessing import format_for_learning, do_grid_search, cross_val, train_test_split
from ..error_quantifiers import rmse, evaluate
from ..plotter import plot_test_predictions, plot_residuals, plot_train_curve, plot_random_forest_feature_importance
from ..models import CustomRandomForestRegressor
from xarray import Dataset
from typing import Optional, Dict, List


def process(
            dataset: Dataset,
            features: List[str],
            n_estimators : Optional[int] = 500,
            max_depth : Optional[int] = 10,
            max_features : Optional[int] = None,
            param_grid: Optional[Dict[str, float]] = None,
            cv_folds: Optional[int] = None,
            plot: Optional[bool] = False,
            base_filepath: Optional[str] = None
):

    # extracts layers from the dataset -> 3D array: H, W, C
    X, y = format_for_learning(dataset, features)

    # splits the dataset into train and test -> 2x 3D arrays: H, W, C
    X_train, X_test, y_train, y_test = train_test_split(X, y, dataset, valid=False)

    model = CustomRandomForestRegressor(n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        max_features=max_features)

    if param_grid:
        model = do_grid_search(CustomRandomForestRegressor(),
                               param_grid, X_train, y_train,
                               cv_folds=cv_folds if cv_folds is not None else 5)
    elif cv_folds is not None:
        cross_val(model, X_train, y_train, cv_folds=cv_folds)

    # Full training
    model.fit(X_train, y_train, is_2D=True)
    y_pred_train = model.predict(X_train, return_2D=True)
    RMSE_train = rmse(y_train.squeeze(), y_pred_train)
    print(f"Final training RMSE: {RMSE_train:.2f} °C")

    # Evaluation
    y_pred_test = model.predict(X_test, return_2D=True)

    result = evaluate(y_pred_test, y_test.squeeze())

    if plot:
        plot_train_curve(model, filepath=f"{base_filepath}_loss.png" if base_filepath else None)
        plot_test_predictions(y_pred_test, dataset, filepath=f"{base_filepath}_predicted_vs_observed.png" if base_filepath else None)
        plot_residuals(y_pred_test, dataset, filepath=f"{base_filepath}_residuals.png" if base_filepath else None)
        plot_random_forest_feature_importance(model, features)

    return model, result