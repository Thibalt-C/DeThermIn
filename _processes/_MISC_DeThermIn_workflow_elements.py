import matplotlib.pyplot as plt
from DeThermIn.data import read_datacube
from owslib.wms import WebMapService
from PIL import Image
import numpy as np
import io
import matplotlib.patches as Rectangle
from DeThermIn.metadata import parameters, extensions, bbox_size
import sys

path = sys.argv[1]

filepath = parameters["processed_data_path"] + "2_3573_48_9222_06_01_2026_08_31_2026"

geostationary_dc = read_datacube(f"{filepath}_{extensions["geostationary"]}.pkl")
narrow_swath_dc = read_datacube(f"{filepath}_{extensions["narrow-swath"]}.pkl")
fusioned_dc = read_datacube(f"{filepath}_{extensions["fusioned_LSTs"]}.pkl")
DeThermIn_dc = read_datacube(f"{filepath}_{extensions["end_product"]}.pkl")
ancillary_dc = read_datacube(f"{filepath}_{extensions['blended_ancillary_data']}.pkl")



narrow_swath = narrow_swath_dc.data.median("time").LST
geostationary = geostationary_dc.data.rio.reproject(geostationary_dc.data.rio.crs).median("time").LST

H, W = geostationary.shape


fig, ax = plt.subplots(figsize=(10, 6))

dx = 5; dy = 3
vmin=10; vmax=35

extent = [dx, dx + W, dy, dy + H]
ax.imshow(geostationary.values, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, alpha=1, extent=extent)
ax.text(extent[1],extent[2], " MTG-FCI LST \n 3 km / 1 h", fontsize=20, ha="left", va="bottom")

extent = [0, W, 0, H]
ax.imshow(narrow_swath.values, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, alpha=1, extent=extent)
ax.text(extent[1],extent[2], " ECOSTRESS LST \n 70 m / 1-5 days", fontsize=20, ha="left", va="bottom")

ax.set_xlim([0, dx + geostationary.shape[1]])
ax.set_ylim([0, dy + geostationary.shape[0]])
ax.axis("off")
plt.title("")
fig.savefig(path+"workflow_element_1.png", bbox_inches="tight", dpi=300)




def add_border(ax, extent, color="white", lw=3):
    ax.add_patch(
        Rectangle.Rectangle(
            (extent[0], extent[1]),
            extent[2] - extent[0],
            extent[3] - extent[1],
            fill=False,
            edgecolor=color,
            linewidth=lw
        )
    )


large_bbox = narrow_swath_dc.data.rio.reproject("EPSG:2154").rio.bounds()
small_bbox = DeThermIn_dc.data.rio.bounds()
plot_extent = [large_bbox[0]-1000, large_bbox[2]+1000, large_bbox[1]-1000, large_bbox[3]+1000]

url = "https://data.geopf.fr/wms-r?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities"
wms = WebMapService(url, version="1.3.0")
layer = "HR.ORTHOIMAGERY.ORTHOPHOTOS"

img = wms.getmap(layers=[layer],
                     size=(4100, 4100),
                     bbox=[plot_extent[0], plot_extent[2], plot_extent[1], plot_extent[3]],
                     srs=f"EPSG:2154",
                     format='image/geotiff')

ortho = Image.open(io.BytesIO(img.read()))
ortho = np.array(ortho)


fig, ax = plt.subplots(figsize=(10,7))
ax.imshow(ortho, extent=plot_extent)
add_border(ax, large_bbox, color="white", lw=3.5)
#add_border(ax, small_bbox, color="white", lw=1)
ax.text(small_bbox[2], small_bbox[1], " Area of\n interest", color="white", fontsize=28)
ax.text(large_bbox[2], large_bbox[1], f" {bbox_size//1000}x{bbox_size//1000} km \n bounding box for LST ", color="white", ha="right", va="bottom", fontsize=28)
DeThermIn_dc.data.mask.where(DeThermIn_dc.data.mask).plot(ax=ax, cmap="grey_r", vmin=1, vmax=2, add_colorbar=False)
ax.set_xlim([plot_extent[0], plot_extent[1]])
ax.set_ylim([plot_extent[2], plot_extent[3]])
ax.axis("off")
ax.set_title("")
plt.tight_layout()
fig.savefig(path+"workflow_element_2.png", bbox_inches="tight", dpi=300)



fig, ax = plt.subplots(figsize=(10, 6))

dx = 300
dy = 50

sel = ancillary_dc.data.sel(sun_elevation=22.5, sun_azimuth=135)

layers = ["SVF", "shadow", "NDVI", "buildings_mask", "DHM"]

for n, layer in enumerate(layers):

    i = len(layers) - n - 1

    im = sel[layer].where(ancillary_dc.data.mask, np.nan)

    extent = [i*dx, i*dx + ancillary_dc.data.rio.shape[1], i*dy, i*dy + ancillary_dc.data.rio.shape[0]]
    ax.imshow(im, cmap=im.plot_kwargs["cmap"], extent=extent)

ax.text(0, 1300, "ancillary data\n2 m", ha="left", fontsize=20)

ax.set_xlim([0, (len(layers)-1)*dx + ancillary_dc.data.rio.shape[1]])
ax.set_ylim([0, (len(layers)-1)*dy + ancillary_dc.data.rio.shape[0]])
ax.axis("off")
plt.title("")
fig.savefig(path+"workflow_element_3.png", bbox_inches="tight", dpi=300)




fusioned = fusioned_dc.data.fine_LST

fig, ax = plt.subplots(figsize=(10, 6))

dx = 3
dy = 3

tot = fusioned.shape[0]

for i in range(tot):

    n = tot - i - 1

    extent = [n*dx, n*dx + fusioned.shape[2], n*dy, n*dy + fusioned.shape[1]]
    ax.imshow(fusioned.isel(time=i).values, cmap="RdYlBu_r", vmin=20, vmax=50, alpha=1, extent=extent)

ax.text(0, -10, "downscaled LST \n70 m / 1 h", fontsize=20, ha="left", va="top")
ax.set_xlim([0, (tot-1)*dx + fusioned.shape[2]])
ax.set_ylim([0, (tot-1)*dy + fusioned.shape[1]])
ax.axis("off")
plt.title("")
fig.savefig(path+"workflow_element_4.png", bbox_inches="tight", dpi=300)


tot = len(DeThermIn_dc.data.time)


ims = DeThermIn_dc.data.SR_LST.where(DeThermIn_dc.data.SR_LST.notnull(), drop=True)
dx = 5; dy = 5
fig, ax = plt.subplots(figsize=(6, 7))
for t in range(tot):
    n = tot - t - 1
    extent = [n * dx, n * dx + ims.values.shape[2], n * dy, n * dy + ims.values.shape[1]]
    ax.imshow(ims.isel(time=t).values,
              cmap="RdYlBu_r", vmin=30, vmax=45,
              extent=extent, alpha=1, aspect="equal")
ax.text((tot-1)*dx + ims.values.shape[1], (tot-1)*dy + ims.values.shape[2], " DeThermIn \n 2 m / 1 h", fontsize=20, ha="left", va="top")
ax.set_xlim([0, (tot-1)*dx + ims.values.shape[2]])
ax.set_ylim([0, (tot-1)*dy + ims.values.shape[1]])
ax.axis("off")
plt.title("")
fig.savefig(path+"workflow_element_5.png", bbox_inches="tight", dpi=300)


fig, ax = plt.subplots(figsize=(10, 6))

dx = 300
dy = 50

sel = DeThermIn_dc.data.sel(sun_elevation=22.5, sun_azimuth=135).isel(time=6)

layers = ["cold_spot_distance", "brightness", "illumination", "irradiance",  "energy_3h"]

for n, layer in enumerate(layers):

    i = len(layers) - n - 1

    im = sel[layer].where(DeThermIn_dc.data.mask, np.nan)

    extent = [i*dx, i*dx + DeThermIn_dc.data.rio.shape[1], i*dy, i*dy + DeThermIn_dc.data.rio.shape[0]]
    ax.imshow(im, cmap=im.plot_kwargs["cmap"], extent=extent)

ax.text(0, 1300, "extra-features\n2 m", ha="left", fontsize=20)

ax.set_xlim([0, (len(layers)-1)*dx + DeThermIn_dc.data.rio.shape[1]])
ax.set_ylim([0, (len(layers)-1)*dy + DeThermIn_dc.data.rio.shape[0]])
ax.axis("off")
plt.title("")
fig.savefig(path+"workflow_element_6.png", bbox_inches="tight", dpi=300)