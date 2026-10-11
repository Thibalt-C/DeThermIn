import matplotlib.patheffects as path_effects
import pickle
import xarray as xr
import matplotlib.pyplot as plt
import multiprocessing
from super_resolution_tools.plotter import add_scale
from sr_workflow import model_path
import os
from copy import deepcopy
import sys

path = sys.argv[1]
processed_data_path = sys.argv[2]
name = sys.argv[3]
extension = sys.argv[4]

def main():

    dataset_path = os.path.join(processed_data_path, f"{name}_{extension}.pkl")
    dataset = pickle.load(open(dataset_path, "rb"))
    small_dataset = deepcopy(dataset)
    small_dataset = small_dataset.rio.clip_box(*(1.43, 43.605, 1.445, 43.615), crs="EPSG:4326")

    static_model = pickle.load(open(f"{model_path}ResUNet_static_model.pkl", "rb"))
    dynamic_model = pickle.load(open(f"{model_path}ResUNet_dynamic_model.pkl", "rb"))


    LST_static = static_model.predict(small_dataset.isel(time=8))
    LST_dynamic = dynamic_model.predict(small_dataset.isel(time=8))

    small_dataset["LST_static"] = xr.DataArray(LST_static, dims=("y", "x"))
    small_dataset["LST_dynamic"] = xr.DataArray(LST_dynamic, dims=("y", "x"))

    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    small_dataset.isel(time=5).energy_1h.plot(ax=axes[0, 0], cmap="jet")
    small_dataset.LST_dynamic.plot(cmap="RdYlBu_r", ax=axes[0, 1],
                                  cbar_kwargs={"label": "Land Surface Temperature °C]"})
    small_dataset.brightness.plot(ax=axes[1, 0], cmap="grey")
    small_dataset.LST_static.plot(cmap="RdYlBu_r", ax=axes[1, 1],
                                 cbar_kwargs={"label": "Land Surface Temperature [°C]"})

    letters = ["a", "b", "c", "d"]

    for ax, letter in zip(axes.flatten(), letters):
        ax.axis("off")
        ax.set_title("")
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

    #add_scale(axes[1,0], LST_dynamic, resol=dataset.rio.resolution()[0], scale_size=50)

    fig.savefig(path + "ResUNet_static_vs_dynamic_models.png", bbox_inches="tight", dpi=300)

if __name__ == "__main__":
    multiprocessing.freeze_support()  # optional but safe on macOS
    main()