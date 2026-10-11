from ..CNNs import extract_patches, ResUNet, DataModule
from ..models import TrainedResUNet
from ..preprocessing import format_for_deep_learning, train_test_split
from ..error_quantifiers import evaluate
from ..plotter import plot_test_predictions, plot_residuals, plot_loss_DL
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from typing import Tuple, Optional, Dict, List
from xarray import Dataset

def process(
        dataset : Dataset,
        features : List[str],
        epochs : int,
        batch_size : int,
        patch_size : int,
        learning_rate : float,
        lambda_ssim : float,
        plot : Optional[bool] = False,
        filepath_base : Optional[str] = None
) -> Tuple[
    TrainedResUNet,
    Dict[str, float]
] :

    # extracts layers from the dataset -> 3D array: H, W, C
    X, y, scaler = format_for_deep_learning(dataset, features)

    # splits the dataset into train, test and validation -> 3x 3D arrays: H, W, C
    X_train, X_test, X_valid, y_train, y_test, y_valid = train_test_split(X, y, dataset)

    # saves the original shape for later evaluation
    true = y_test.squeeze()

    # Patch extraction -> 3x 4D arrays: N_patches, H, W, C
    stride = patch_size // 2 # cannot be changed!
    X_train, y_train, valid_train_patches = extract_patches(X_train, y_train, patch_size=patch_size, stride=stride)
    X_valid, y_valid, valid_train_patches = extract_patches(X_valid, y_valid, patch_size=patch_size, stride=stride)
    X_test, y_test, valid_test_patches = extract_patches(X_test, y_test, patch_size=patch_size, stride=stride)


    # sets up the data module for the deep learning model
    dm = DataModule(
        X_train, X_valid, X_test,
        y_train, y_valid, y_test,
        batch_size=batch_size)
    dm.setup()

    # sets up the deep learning model
    model = ResUNet(
        in_channels=len(features),
        out_channels=1,
        lr=learning_rate,
        lambda_ssim=lambda_ssim)

    # prepares the trainer with a fixed number of epochs + progress bar
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        callbacks=[RichProgressBar()])

    # fits the model with the data loaded
    trainer.fit(model, dm)

    # sets the model in evaluation mode
    model.eval()

    # creates the model as capable of handling Dataset objects for further predictions
    trained_model = TrainedResUNet(model, features, batch_size, epochs, patch_size, scaler)

    # predicts the test set and evaluates the model in terms of metrics
    dataset_test = dataset.where(dataset.test_mask)
    pred = trained_model.predict(dataset_test)
    result = evaluate(pred, true)

    # makes plots if requested
    if plot:
        plot_loss_DL(model.train_losses,
                     model.val_losses,
                     len(dm.train_dataloader()),
                     len(dm.val_dataloader()),
                     filepath=f"{filepath_base}_loss.png"
                     )
        plot_test_predictions(pred,
                              dataset_test,
                              filepath=f"{filepath_base}_predicted_vs_observed.png"
                              )
        plot_residuals(pred,
                       dataset_test,
                       filepath=f"{filepath_base}_residuals.png"
                       )

    return trained_model, result