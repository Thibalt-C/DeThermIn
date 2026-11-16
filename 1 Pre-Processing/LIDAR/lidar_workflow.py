import pickle
import sys
import json
import time
import pandas as pd
import numpy as np
import os

gdf_pkl = sys.argv[1]
name = sys.argv[2]
resolution = sys.argv[3]
processed_lidar_folder = sys.argv[4]
extension = sys.argv[5]
country = sys.argv[6]
if sys.argv[7] != "None":
    time_axis = pd.to_datetime(sys.argv[7].replace("\n","").replace("'","").replace("[","").replace("]","").split(" "))
else:
    time_axis = None

base_resol = 0.5
decim = int(float(resolution) // base_resol)
actual_resolution = decim * base_resol
print(f"Computing for a spatial resolution of {actual_resolution:.2f} m.")
print(f"Country: {country}")

time.sleep(2); gdf = pickle.load(open(gdf_pkl, 'rb'))

with open("folders.json", "r") as f:
    folders = json.load(f)

lidar_decim = 10
build = True
layer_for_derivation = "DSM"
threshold_for_warning = np.inf
shadow_azimuth_resolution = 22.5
shadow_elevation_resolution = 10.0
verbose = True


if country == "FRA":

    import lidar_hd_tools as lhd  # (FRA) dedicated PyPI package

    download_func = lhd.download_data
    download_params = [gdf, decim, lidar_decim, build, layer_for_derivation, time_axis, shadow_azimuth_resolution, shadow_elevation_resolution,  threshold_for_warning, verbose]

    buildings_func = lhd.get_buildings_mask
    land_cover_func = lhd.get_land_occupation
    orthoimage_func = lhd.get_orthoimage

    lhd.change_folder("DEM", folders["DEM"])
    lhd.change_folder("DSM", folders["DSM"])
    lhd.change_folder("lidar", folders["lidar"])


elif country == "NLD":

    with open("ahn_tools/folders.json", "w") as f:
        json.dump(folders, f, indent=4)
    time.sleep(3)
    import ahn_tools as ahn

    download_func = ahn.download_data
    download_params = [gdf, decim, lidar_decim, build, layer_for_derivation, threshold_for_warning, verbose]

    buildings_func = None
    land_cover_func = ahn.get_land_cover
    orthoimage_func = ahn.get_orthoimage

# ... would be easy to add other airborne LiDAR data with different parsing methods!


lidar_dataset = download_func(*download_params)

# in case there is errors, to save
pickle.dump(lidar_dataset, open(f"{processed_lidar_folder}{name}_{extension}_uncomplete.pkl", "wb"))

if time_axis is not None:
    lidar_dataset = lidar_dataset.rename_dims({"time": "shadow_time"})
    lidar_dataset = lidar_dataset.rename({"time": "shadow_time"})

lidar_dataset = buildings_func(lidar_dataset) if buildings_func is not None else lidar_dataset
print("Getting land cover data...")
retry=0
while retry<=10:
    try:
        lidar_dataset = land_cover_func(lidar_dataset) if land_cover_func is not None else lidar_dataset
        retry=10
    except:
        retry +=1
print("Getting ortho-image...")
retry=0
while retry<=10:
    try:
        lidar_dataset = orthoimage_func(lidar_dataset) if orthoimage_func is not None else lidar_dataset
        retry=10
    except:
        retry +=1

pickle.dump(lidar_dataset, open(f"{processed_lidar_folder}{name}_{extension}.pkl", "wb"))
os.remove(f"{processed_lidar_folder}{name}_{extension}_uncomplete.pkl") # remove the uncomplete file

print("-> LiDAR workflow is done.")