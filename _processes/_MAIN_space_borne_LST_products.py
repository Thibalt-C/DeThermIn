from DeThermIn.data import read_datacube
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from DeThermIn.data.visualisation import add_scale
from DeThermIn.metadata import parameters, extensions
import os
import sys

path = sys.argv[1]

name = "2_3573_48_9222_06_01_2026_08_31_2026"
geostationary_path = os.path.join(parameters["processed_data_path"], f"{name}_{extensions['geostationary']}.pkl")
narrow_swath_path = os.path.join(parameters["processed_data_path"], f"{name}_{extensions['narrow-swath']}.pkl")

geostationary_dc = read_datacube(geostationary_path)
narrow_swath_dc = read_datacube(narrow_swath_path)

point_zero = [narrow_swath_dc.data.y.mean().values, narrow_swath_dc.data.x.mean().values]


geostationary = geostationary_dc.data.rio.reproject_match(narrow_swath_dc.data).median("time").LST
narrow_swath = narrow_swath_dc.data.median("time").LST

x = len(narrow_swath_dc.data.x)
split_idx = x // 2

fig, ax = plt.subplots(figsize=(6,5))
narrow_swath.where(narrow_swath.x<=narrow_swath_dc.data.x[split_idx]).plot(ax=ax, cmap="RdYlBu_r", add_colorbar=False, vmin=15, vmax=50)
quadmesh = geostationary.where(geostationary.x>narrow_swath_dc.data.x[split_idx]).plot(ax=ax, cmap="RdYlBu_r", vmin=15, vmax=50)
ax.scatter(point_zero[1], point_zero[0], marker="+", color="w", s=100)
quadmesh.colorbar.set_label("Land Surface Temperature [°C]", fontsize=15, labelpad=-10)
quadmesh.colorbar.set_ticks([15, 50])
quadmesh.colorbar.ax.tick_params(labelsize=12)
add_scale(ax, narrow_swath_dc, scale_size=5000)
ax.set_title("")
ax.axis("off")
plt.tight_layout()

fig.savefig(path+"used_LST_products_spatial.png", bbox_inches="tight", dpi=300)






begin = datetime.date(2026, 8, 10)
n = 4
end = begin + datetime.timedelta(days=n)


geostationary = geostationary_dc.data.rio.reproject_match(narrow_swath_dc.data).where((geostationary_dc.data.time.dt.date>=begin) & (geostationary_dc.data.time.dt.date<=end), drop=True).LST
narrow_swath = narrow_swath_dc.data.LST.where((narrow_swath_dc.data.time.dt.date>=begin) & (narrow_swath_dc.data.time.dt.date<=end), drop=True)

fig, ax = plt.subplots(figsize=(8,5))
geostationary.sel(x=point_zero[1], y=point_zero[0], method="nearest").plot(c="orange", label="MTG-FCI", x="time", lw=2)
narrow_swath.sel(x=point_zero[1], y=point_zero[0], method="nearest").plot.scatter(marker="*", c="g", label="ECOSTRESS", x="time", s=400)
ax.legend(fontsize=16)
ax.grid(True)
ax.set_xlim([begin, end])
ax.set_ylim([15, 50])
ax.set_ylabel("")
ax.set_xlabel("")
ax.set_title("")
ax.set_xticks(pd.date_range(start=begin, end=end, freq="1D"),
              [(begin+datetime.timedelta(days=i)).strftime("%d %b") for i in range(n+1)])
ax.tick_params(labelsize=12)

plt.tight_layout()

fig.savefig(path+"used_LST_products_temporal.png", bbox_inches="tight", dpi=300)

