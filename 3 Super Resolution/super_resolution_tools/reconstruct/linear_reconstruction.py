from sklearn.model_selection import train_test_split
from ..preprocessing import format_for_learning, cross_val, do_grid_search
from ..error_quantifiers import evaluate, rmse
from ..plotter import plot_residuals, plot_train_curve, plot_test_predictions
from ..models import CustomLinearRegression


def process(dataset, features, plot=False, save_fig=False, filepath=None,
            train_size=0.8,
            learning_rate=0.1, max_iterations=1000, step_threshold=1e-6,
            cv_folds=None, grid_search=False, param_grid=None):

    X, y, nan_mask = format_for_learning(dataset, features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, shuffle=False)

    model = CustomLinearRegression(learning_rate=learning_rate,
                                   max_iterations=max_iterations,
                                   step_threshold=step_threshold)

    # Cross validation if activated
    if grid_search and param_grid:
        model = do_grid_search(CustomLinearRegression(),
                               param_grid, X_train, y_train,
                               cv_folds=cv_folds if cv_folds is not None else 5)
    elif cv_folds is not None:
        cross_val(model, X_train, y_train, cv_folds=cv_folds)

    # Full training
    model.fit(X_train, y_train.squeeze())
    y_pred_train = model.predict(X_train)
    RMSE_train = rmse(y_train.squeeze(), y_pred_train)
    print(f"Final training RMSE: {RMSE_train:.2f} °C")

    # Evaluation
    y_pred_test = model.predict(X_test)

    result = evaluate(y_pred_test, y_test)

    if plot:
        plot_train_curve(model)
        plot_test_predictions(y_pred_test, dataset, nan_mask, train_size, save_fig, filepath)
        plot_residuals(y_pred_test, dataset, nan_mask, train_size)

    return model, result