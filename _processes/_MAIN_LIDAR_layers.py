from copy import deepcopy
from DeThermIn.data import read_datacube
from DeThermIn.data.visualisation import add_scale
# noinspection PyUnusedImports
import rioxarray
import matplotlib.pyplot as plt
import os
from DeThermIn.metadata import parameters, extensions
import sys

path = sys.argv[1]

name = "1_4445_43_6085_05_01_2021_07_30_2021"
dc_path = os.path.join(parameters["processed_data_path"], f"{name}_{extensions['enriched_ancillary_data']}.pkl")
dc_enriched = read_datacube(dc_path)

small_dc = deepcopy(dc_enriched)
small_dc.data = small_dc.data.rio.clip_box(*(1.43,43.605,1.445,43.615), crs="EPSG:4326")

fig, axes = plt.subplots(1,3, figsize=(14,4.5))

layer = small_dc.data.water_mask
layer.where(layer).plot(cmap=layer.plot_kwargs["cmap"], add_colorbar=False, ax=axes[0])
layer = small_dc.data.vegetalized
layer.where(layer).plot(cmap=layer.plot_kwargs["cmap"], add_colorbar=False, ax=axes[0])
layer = small_dc.data.artificial
layer.where(layer).plot(cmap=layer.plot_kwargs["cmap"], add_colorbar=False, ax=axes[0])
layer = small_dc.data.mineralized
layer.where(layer).plot(cmap=layer.plot_kwargs["cmap"], add_colorbar=False, ax=axes[0])
layer = small_dc.data.buildings_mask
layer.where(layer).plot(cmap=layer.plot_kwargs["cmap"], add_colorbar=False, ax=axes[0])
axes[0].axis("off")
axes[0].set_title("Land-cover and buildings mask", fontsize=16, pad=-20)

layer = small_dc.data.orthoimage
layer.where(layer).plot.imshow(ax=axes[1], robust=True)
axes[1].axis("off")
axes[1].set_title("Ortho-image", fontsize=16)


layer = small_dc.data.SVF
layer.where(layer).plot.imshow(cmap=layer.plot_kwargs["cmap"], add_colorbar=False, ax=axes[2])
#layer = small_dc.data.DSM
#layer.where(layer).where(small_dc.data.buildings_mask).plot.imshow(cmap=layer.plot_kwargs["cmap"], add_colorbar=False, ax=axes[2])
layer = small_dc.data.vegetation
layer.where(layer).plot.imshow(cmap=layer.plot_kwargs["cmap"], vmax=1, add_colorbar=False, ax=axes[2])
axes[2].axis("off")
axes[2].set_title("SVF and vegetation cover", fontsize=16)


add_scale(axes[0], small_dc, scale_size=100)


plt.tight_layout()
fig.savefig(path + "LIDAR_derived_layers.png", bbox_inches="tight", dpi=300)