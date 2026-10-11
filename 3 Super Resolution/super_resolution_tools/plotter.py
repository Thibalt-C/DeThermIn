import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from .error_quantifiers import rmse
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects


def plot_test_predictions(pred, dataset, filepath=None):

    dataset["pred"] = xr.DataArray(pred, dims=("y", "x"), coords={"y": dataset.y, "x": dataset.x})
    dataset = dataset.where(dataset.pred.notnull(), drop=True)

    true = dataset.LST_ground.values
    pred = dataset.pred.values
    residuals = true - pred
    coarse_LST = dataset.fine_LST

    fig = plt.figure(figsize=(10, 16), dpi=400)
    gs = gridspec.GridSpec(
        4, 1,
        figure=fig,
        hspace=0.01
    )

    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2]), fig.add_subplot(gs[3])]
    axes[0].imshow(coarse_LST, cmap='RdYlBu_r', vmin=np.nanmin(true), vmax=np.nanmax(true))
    axes[1].imshow(true, cmap='RdYlBu_r', vmin=np.nanmin(true), vmax=np.nanmax(true))
    im1=axes[2].imshow(pred, cmap='RdYlBu_r', vmin=np.nanmin(true), vmax=np.nanmax(true))
    im2=axes[3].imshow(residuals, cmap='RdBu_r', vmin=-np.nanmax(abs(residuals)), vmax=np.nanmax(abs(residuals)))

    add_scale(axes[3], pred, resol=2)

    cbar_ax1 = fig.add_axes([0.3, 0.09, 0.4, 0.009])  # top colorbar
    cbar_ax2 = fig.add_axes([0.3, 0.01, 0.4, 0.009])  # bottom colorbar
    fig.colorbar(im1, cax=cbar_ax1, orientation='horizontal',
                 label="Land Surface Temperature [°C]")
    fig.colorbar(im2, cax=cbar_ax2, orientation='horizontal',
                 label="Residuals (model - ground truth) [°C]")

    letters = ["a", "b", "c", "d"]

    for ax, letter in zip(axes, letters):
        ax.axis("off")
        txt = ax.text(
            0.02, 0.95, letter,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
            color="white"
        )
        txt.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground="black"),
            path_effects.Normal()
        ])

    if filepath is not None:
        fig.savefig(filepath, bbox_inches="tight", dpi=300)
    else:
        plt.show()


def add_scale(ax, y_pred_test, resol, scale_size=100):
    H, W = y_pred_test.shape
    n_pixels = int(scale_size // resol)

    x = np.arange(W)
    y = np.arange(H)

    center_x = 50 - W
    lower_y = 50 - H

    x_idxs = [center_x, center_x + n_pixels]
    y_idxs = [-lower_y, -lower_y]

    line = ax.plot(x[x_idxs], y[y_idxs], "-", color="w", linewidth=10)
    text = ax.text(
        x[x_idxs[1]], y[y_idxs[1]],
        f"   {scale_size} m",
        ha="left", va="center", fontweight="bold", color="w", fontsize=12
    )

    text.set_path_effects([
        path_effects.Stroke(linewidth=2, foreground="black"),
        path_effects.Normal()
    ])
    line[0].set_path_effects([
        path_effects.Stroke(linewidth=13, foreground="black"),
        path_effects.Normal()
    ])



def plot_train_curve(lin_model, filepath=None):
    fig=plt.figure()
    plt.plot(lin_model.train_errors[1:], '-k', lw=2, label="Training fold")
    plt.xlabel("Iteration")
    plt.ylabel("Relative error")
    plt.legend()
    if filepath is not None:
        fig.savefig(filepath, bbox_inches="tight", dpi=300)
    else:
        plt.show()


def plot_residuals(pred, dataset, filepath=None):

    true = dataset.LST_ground.values
    residuals = true - pred

    resid_std = np.nanstd(residuals)

    bias = np.nanmean(residuals)
    RMSE = rmse(pred, true)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax2 = axes[0].twiny()

    axes[0].hist(residuals.flatten(),
                 bins=np.linspace(-4 * resid_std, 4 * resid_std, 200),
                 color="k")

    ax2.set_xticks(
        [-4 * resid_std, -3 * resid_std, -2 * resid_std, -resid_std, 0, resid_std, 2 * resid_std, 3 * resid_std,
         4 * resid_std],
        [r"$-4\sigma$", r"$-3\sigma$", r"$-2\sigma$", r"$-\sigma$", 0, r"$\sigma$", r"$2\sigma$", r"$3\sigma$",
         r"$4\sigma$"])

    axes[0].set_xlabel("Residuals [°C]")
    axes[0].set_ylabel("# of pixels")

    axes[1].set_xlabel("Super-Resolution LST [°C]")
    axes[1].set_ylabel("Ground truth LST [°C]")

    axes[0].legend(title=f"RMSE: {RMSE:.2f} °C\nbias: {bias:.2f} °C")
    axes[1].scatter(pred.flatten(), true.flatten(), color="k", s=1, alpha=0.01)
    axes[1].set_xlim([10, 80])
    axes[1].set_ylim([10, 80])
    axes[1].plot(np.linspace(10, 80, 3), np.linspace(10, 80, 3), c="k", ls="--")

    if filepath is not None:
        fig.savefig(filepath, bbox_inches="tight", dpi=300)
    else:
        plt.show()


def plot_loss_DL(train_losses, val_losses, train_batches_per_epoch, val_batches_per_epoch, filepath=None):

        train_epoch_loss = [
            np.mean(train_losses[i * train_batches_per_epoch:(i + 1) * train_batches_per_epoch])
            for i in range(len(train_losses) // train_batches_per_epoch)
        ]

        val_epoch_loss = [
            np.mean(val_losses[i * val_batches_per_epoch:(i + 1) * val_batches_per_epoch])
            for i in range(len(val_losses) // val_batches_per_epoch)
        ]

        train_epochs = range(1, len(train_epoch_loss) + 1)
        val_epochs = range(1, len(val_epoch_loss) + 1)

        fig = plt.figure(figsize=(10, 6))
        plt.plot(train_epochs, train_epoch_loss, label="Train Loss", linewidth=2)
        plt.plot(val_epochs, val_epoch_loss, label="Validation Loss", linewidth=2)

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        if filepath is not None:
            fig.savefig(filepath, bbox_inches="tight", dpi=300)
        else:
            plt.show()


def plot_random_forest_feature_importance(model, features):
    fig, ax = plt.subplots(figsize=(12,4))
    plt.bar(features, height=model.regressor.feature_importances_*100, color="k")
    ax.set_xticks(features, features, rotation=45, ha="right")
    ax.set_ylabel("Importance [%]")
    plt.tight_layout()
    plt.show()