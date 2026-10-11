import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from DeThermIn.data import read_datacube
from DeThermIn.metadata import parameters, extensions
import os
import sys

path = sys.argv[1]

name = "2_3573_48_9222_06_01_2026_08_31_2026"

ancillary_path = os.path.join(parameters["processed_data_path"], f"{name}_{extensions["blended_ancillary_data"]}.pkl")
hsr_path = os.path.join(parameters["processed_data_path"], f"{name}_{extensions["multispectral"]}.pkl")

ancillary_dc = read_datacube(ancillary_path)
hsr_dc = read_datacube(hsr_path)

hsr_dc.data = hsr_dc.data.rio.clip_box(*ancillary_dc.data.rio.bounds(), crs=ancillary_dc.data.rio.crs)

fig, axes = plt.subplots(1,2, figsize=(14,4), subplot_kw={"projection": ccrs.epsg(32631)})
_, im1 = hsr_dc.plot("NDVI",gridlines=False, ax=axes[0], cbar_kwargs={"shrink":0.85})
_, im2 = hsr_dc.plot("ENDISI",gridlines=False, ax=axes[1], cbar_kwargs={"shrink":0.85})


for im, title_top, title_bottom in zip([im1, im2], ["Rich vegetation", "Paved/impervious"], ["Poor vegetation", "Permeable"]):

    cbar = im.colorbar

    cbar.ax.text(
        0, 1.05, title_top,
        ha="left", va="bottom", transform=cbar.ax.transAxes
    )

    cbar.ax.text(
        0, -0.05, title_bottom,
        ha="left", va="top", transform=cbar.ax.transAxes
    )

for ax in axes:
    ax.set_title("")
fig.savefig(path+"sentinel2_multispectral_indices.png", bbox_inches="tight", dpi=300)