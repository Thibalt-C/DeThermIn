import subprocess
import os
import pandas as pd
import rioxarray as rxr
import datetime
import sys
import pickle
import numpy as np
from rasterio.enums import Resampling
from shapely.geometry import shape
from rasterio.features import shapes
import geopandas as gpd

target_resolution = float(sys.argv[1])
processed_data_path = sys.argv[2]
time_extent = int(sys.argv[3])
extension = sys.argv[4]

filename = "mosaic_centre_LST.tif"
downloading_script = "download_ground_truth.py"

if not os.path.exists(filename):

    process = subprocess.Popen(["python", downloading_script])
    process.wait()


ground_truth = rxr.open_rasterio(filename)

ground_truth = ground_truth.to_dataset(name="LST_ground")
ground_truth["mask"] = ground_truth.LST_ground != 0
ground_truth["mask"] = ground_truth.mask.astype(np.uint8)
ground_truth = ground_truth.where(ground_truth.mask, drop=True).squeeze()
ground_truth.LST_ground.data -= 273.15

ground_truth.mask.attrs = {"standard_name": "Geographic mask",
                           "units": "no units",
                           "plot_kwargs": {"cmap": "grey", "vmin":0, "vmax":1}
                           }

ground_truth.coords["time"] = np.array("2021-06-15 11:00:00", dtype="datetime64[ns]")
ground_truth.LST_ground.attrs["original_resolution"] = ground_truth.rio.resolution()[0]
ground_truth.LST_ground.attrs["units"] = "°C"
ground_truth.LST_ground.attrs["plot_kwargs"] = {"cmap":"RdYlBu_r"}
ground_truth.LST_ground.attrs["standard_name"] = "Airborne-based Land surface temperature"
ground_truth.LST_ground.attrs["description"] = "Multispectral thermal infrared imagery was acquired with a Telops camera equipped \
with a filter wheel in 5 spectral bands. The data delivered are the brightness temperature at sensor level, the \
radiance at sensor level and the Land Surface Temperature (LST)."
ground_truth.LST_ground.attrs["doi"] = "https://doi.org/10.6096/1012"
ground_truth.LST_ground.attrs["cite"] = "Barillot, P. (2022). Airborne thermal infrared data. [dataset]. Theia."

ground_truth = ground_truth.rio.reproject(dst_crs="EPSG:2154",
                                          resolution=target_resolution,
                                          resampling=Resampling.bilinear)

geoms = [
    shape(geom)
    for geom, value in shapes(
        ground_truth.mask.astype("uint8").values,
        mask=ground_truth.mask.astype("uint8").values,
        transform=ground_truth.rio.transform()
    )
    if value == 1
]

gdf = gpd.GeoDataFrame(geometry=geoms, crs=ground_truth.rio.crs).to_crs("EPSG:4326")

lon, lat = gdf.union_all().centroid.coords[0]

start = (pd.to_datetime(ground_truth.time.values) - datetime.timedelta(days=time_extent//2)).strftime("%m-%d-%Y")
end = (pd.to_datetime(ground_truth.time.values) + datetime.timedelta(days=time_extent//2)).strftime("%m-%d-%Y")

coords_str = (f"{round(lon, 4)}_{round(lat, 4)}").replace('.','_')
filename = coords_str + '_' + start.replace('-', '_') + '_' + end.replace('-', '_') + "_" + extension + ".pkl"

filepath = f"{processed_data_path}{filename}"
pickle.dump(ground_truth, open(filepath, "wb"))

print(f"-> Created {filepath}")
