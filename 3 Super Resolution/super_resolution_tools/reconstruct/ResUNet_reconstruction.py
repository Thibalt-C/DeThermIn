from ..CNNs import extract_patches, reconstruct_DL, ResUNet, DataModule
from ..models import trained_UNet
from ..preprocessing import format_for_deep_learning
from ..error_quantifiers import evaluate
from ..plotter import plot_test_predictions, plot_residuals, plot_loss_DL
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl

def process(dataset, features,
            train_size, valid_size,
            epochs, batch_size, patch_size,
            learning_rate, lambda_ssim,
            plot=False, save_fig=False, filepath=None):

    X, y, scaler = format_for_deep_learning(dataset, features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1-train_size, shuffle=False)

    # Save original shape for reconstruction
    H, W = y_test.shape[:2]
    test = y_test.squeeze()

    # Patch extraction
    stride = patch_size // 2
    X_train, y_train, valid_train_patches = extract_patches(
        X_train, y_train, patch_size=patch_size, stride=stride)
    X_test, y_test, valid_test_patches = extract_patches(
        X_test, y_test, patch_size=patch_size, stride=stride)

    X_val, X_train, y_val, y_train = train_test_split(
        X_train, y_train, test_size=1-valid_size, shuffle=False)

    dm = DataModule(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        batch_size=batch_size)
    dm.setup()

    model = ResUNet(
        in_channels=len(features),
        out_channels=1,
        lr=learning_rate,
        lambda_ssim=lambda_ssim)

    trainer = pl.Trainer(
        max_epochs=epochs,
        log_every_n_steps=1,
        enable_checkpointing=False)
    trainer.fit(model, dm)

    model.eval()
    y_pred_test = reconstruct_DL(
        model, dm.test_dataloader(),
        patch_size, stride, H, W,
        valid_test_patches)

    result = evaluate(y_pred_test.flatten(), test.flatten())

    if plot:
        plot_loss_DL(model.train_losses, model.val_losses, len(dm.train_dataloader()), len(dm.val_dataloader()))
        plot_test_predictions(y_pred_test, dataset, save=save_fig, filepath=filepath)
        plot_residuals(y_pred_test, dataset)

    trained_model = trained_UNet(model, features, batch_size, epochs, patch_size, scaler)

    return trained_model, result