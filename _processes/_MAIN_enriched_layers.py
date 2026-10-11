from copy import deepcopy
import pandas as pd
from DeThermIn.data import read_datacube
from DeThermIn.data.visualisation import add_scale
# noinspection PyUnusedImports
import rioxarray
import matplotlib.pyplot as plt
from DeThermIn.metadata import parameters, extensions
import os
import sys

path = sys.argv[1]

name = "1_4445_43_6085_05_01_2021_07_30_2021"
dc_path = os.path.join(parameters["processed_data_path"], f"{name}_{extensions['enriched_ancillary_data']}.pkl")
dc_enriched = read_datacube(dc_path)

small_dc = deepcopy(dc_enriched)
small_dc.data = small_dc.data.rio.clip_box(*(1.43,43.605,1.445,43.615), crs="EPSG:4326")

fig, axes = plt.subplots(2,3, figsize=(16,7))

for t in range(1,4):

    layer = small_dc.data.isel(time=t*3).irradiance
    im = layer.plot(cmap=layer.plot_kwargs["cmap"], ax=axes[0,t-1])
    axes[0,t-1].axis("off")
    axes[0,t-1].set_title(pd.to_datetime(layer.time.values).strftime("%H:%M UTC"), fontsize=18)
    im.colorbar.set_ticks([layer.min(), layer.max()])
    im.colorbar.set_label(layer.attrs["standard_name"] + f" [{layer.attrs["units"]}]", fontsize=14, labelpad=-10)

    layer = small_dc.data.isel(time=t*3).energy
    im = layer.plot(cmap=layer.plot_kwargs["cmap"], ax=axes[1,t-1])
    axes[1,t-1].axis("off")
    axes[1,t-1].set_title("")
    im.colorbar.set_ticks([layer.min(), layer.max()])
    im.colorbar.set_label(layer.attrs["standard_name"] + f" [{layer.attrs["units"]}]", fontsize=14, labelpad=-10)

plt.tight_layout()
add_scale(axes[1,0], small_dc, scale_size=200)
fig.savefig(f"{path}enriched_layers_dynamic.png", bbox_inches="tight", dpi=300)



fig, axes = plt.subplots(1,2, figsize=(10,4))

layer = small_dc.data.isel(time=0).cold_spot_distance
im = layer.plot(cmap=layer.plot_kwargs["cmap"], ax=axes[0])
axes[0].set_title("Cold spot distance", fontsize=12)
im.colorbar.set_ticks([layer.min(), layer.max()])
im.colorbar.set_label(layer.attrs["standard_name"] + f" [{layer.attrs["units"]}]", fontsize=12, labelpad=-10)

layer = small_dc.data.isel(time=0).illumination
im = layer.plot(cmap=layer.plot_kwargs["cmap"], ax=axes[1])
axes[1].set_title("Daily illumination", fontsize=12)
im.colorbar.set_ticks([layer.min(), layer.max()])
im.colorbar.set_label(layer.attrs["standard_name"] + f" [{layer.attrs["units"]}]", fontsize=12, labelpad=-10)

add_scale(axes[0], small_dc, scale_size=200)

for ax in axes:
    ax.axis("off")
plt.tight_layout()
fig.savefig(f"{path}enriched_layers_static.png", bbox_inches="tight", dpi=300)