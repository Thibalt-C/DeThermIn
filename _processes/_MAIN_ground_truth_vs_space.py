import matplotlib.pyplot as plt
import pickle
import pandas as pd
import DeThermIn.data.visualisation as vis
from DeThermIn.workflows.workflow_manager import *
import datetime
from DeThermIn.data import read_datacube
from DeThermIn.metadata import extensions, parameters
from DeThermIn.data.visualisation import add_scale
import os
import sys

path = sys.argv[1]

name = "1_4445_43_6085_05_01_2021_07_30_2021"
path1 = os.path.join(parameters["processed_data_path"], f"{name}_{extensions["training_data"]}.pkl")
path2 = os.path.join(parameters["processed_data_path"], f"{name}_{extensions["narrow-swath"]}.pkl")

dc = read_datacube(path1)
dc2 = read_datacube(path2)

dc2.data = dc2.data.rio.clip_box(*dc.data.rio.bounds(), crs=dc.data.rio.crs).rio.reproject_match(dc.data)
selection = dc2.data.sel(time=datetime.datetime(2021,6,9, 11), method="nearest")
selection = selection.where(dc.data.mask)

fig, axs = plt.subplots(1,2, figsize=(12, 5))

im1 = selection.LST.plot(ax=axs[0], cmap="RdYlBu_r")
axs[0].set_title(f"ECOSTRESS acquisition \n({pd.to_datetime(selection.time.values).strftime('%d-%m-%Y %H:%M UTC')})", fontsize=14)
im2 = dc.data.LST_ground.plot(ax=axs[1], cmap="RdYlBu_r")
axs[1].set_title("Airborne acquisition \n(15-06-2021 around 11:00 UTC)", fontsize=14)

for im, LST in zip([im1, im2],[selection.LST, dc.data.LST_ground]):
    im.colorbar.set_label("Land Surface Temperature [°C]", fontsize=14, labelpad=-20)
    im.colorbar.set_ticks([LST.min(), LST.max()])

add_scale(axs[0], dc2, scale_size=500)

for ax in axs:
    ax.axis("off")

plt.tight_layout()

fig.savefig(path+"ground_truth_vs_space.png", bbox_inches='tight', dpi=300)