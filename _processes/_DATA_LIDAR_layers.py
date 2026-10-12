from copy import deepcopy
from DeThermIn.data import read_datacube
from DeThermIn.data.visualisation import add_scale
# noinspection PyUnusedImports
import rioxarray
import matplotlib.pyplot as plt
import os
from DeThermIn.metadata import parameters, extensions
import sys
from matplotlib.patches import Patch

path = sys.argv[1]
name = sys.argv[2]

dc_path = os.path.join(parameters["processed_data_path"], f"{name}_{extensions['enriched_ancillary_data']}.pkl")
dc_enriched = read_datacube(dc_path)

small_dc = deepcopy(dc_enriched)
small_dc.data = small_dc.data.rio.clip_box(*(2.35,48.919,2.365,48.928), crs="EPSG:4326")

fig, axes = plt.subplots(2,2, figsize=(10,8))

axes = axes.ravel()

layers = [
    ("Water", small_dc.data.water_mask),
    ("Vegetation", small_dc.data.vegetalized),
    ("Artificial", small_dc.data.artificial),
    ("Mineralized", small_dc.data.mineralized),
    ("Buildings", small_dc.data.buildings_mask),
]

handles = []

for label, layer in layers:
    cmap = plt.get_cmap(layer.plot_kwargs["cmap"])
    layer.where(layer).plot.imshow(cmap=cmap, ax=axes[0], add_colorbar=False)
    handles.append(
        Patch(
            facecolor=cmap(0.5),  # ou cmap(1.0)
            edgecolor="black",
            label=label,
        )
    )

axes[0].legend(handles=handles)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].set_xlabel("")
axes[0].set_ylabel("")
axes[0].set_title("Land cover", fontsize=16)

layer = small_dc.data.orthoimage
layer.where(layer).plot.imshow(ax=axes[1], robust=True)
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].set_xlabel("")
axes[1].set_ylabel("")
axes[1].set_title("Ortho-image", fontsize=16)


layer = small_dc.data.SVF
mesh = layer.where(layer).plot.imshow(cmap=layer.plot_kwargs["cmap"], ax=axes[2], add_colorbar=False)
#layer = small_dc.data.DSM
#layer.where(layer).where(small_dc.data.buildings_mask).plot.imshow(cmap=layer.plot_kwargs["cmap"], add_colorbar=False, ax=axes[2])
axes[2].set_xticks([])
axes[2].set_yticks([])
axes[2].set_xlabel("")
axes[2].set_ylabel("")
axes[2].set_title("Sky View Factor", fontsize=16)
pos = axes[2].get_position()
cax = fig.add_axes([pos.x0,pos.y0-0.05,pos.width,0.02])
fig.colorbar(mesh, cax=cax, orientation="horizontal", label= "Sky View Factor [no units]")


add_scale(axes[2], small_dc, scale_size=100)

layer = small_dc.data.vegetation
mesh = layer.where(layer).plot.imshow(cmap=layer.plot_kwargs["cmap"], ax=axes[3], add_colorbar=False)
axes[3].set_xticks([])
axes[3].set_yticks([])
axes[3].set_xlabel("")
axes[3].set_ylabel("")
axes[3].set_title("Vegetation density", fontsize=16)
pos = axes[3].get_position()
cax = fig.add_axes([pos.x0,pos.y0-0.05,pos.width,0.02])
fig.colorbar(mesh, cax=cax, orientation="horizontal", label= "Vegetation density [pt/pixel]")

fig.subplots_adjust(hspace=0.15, wspace=0.1)

#plt.tight_layout()
fig.savefig(path + "LIDAR_derived_layers.png", bbox_inches="tight", dpi=300)