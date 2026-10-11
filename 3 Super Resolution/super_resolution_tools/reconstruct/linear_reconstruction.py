from ..preprocessing import format_for_learning, cross_val, do_grid_search, train_test_split
from ..error_quantifiers import evaluate, rmse
from ..plotter import plot_residuals, plot_train_curve, plot_test_predictions
from ..models import CustomLinearRegression
from xarray import Dataset
from typing import Tuple, Optional, Dict, List


def process(
        dataset : Dataset,
        features : List[str],
        learning_rate : Optional[float] = 0.1,
        max_iterations : Optional[int] = 1000,
        step_threshold : Optional[float] = 1e-6,
        param_grid : Optional[Dict[str, float]] = None,
        cv_folds : Optional[int] = None,
        plot : Optional[bool] = False,
        base_filepath : Optional[str] = None
):

    # extracts layers from the dataset -> 3D array: H, W, C
    X, y = format_for_learning(dataset, features)

    # splits the dataset into train and test -> 2x 3D arrays: H, W, C
    X_train, X_test, y_train, y_test = train_test_split(X, y, dataset, valid=False)

    # initializes the model
    model = CustomLinearRegression(learning_rate=learning_rate,
                                   max_iterations=max_iterations,
                                   step_threshold=step_threshold)

    # Cross validation if activated
    if param_grid:
        model = do_grid_search(CustomLinearRegression(),
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

    return model, result