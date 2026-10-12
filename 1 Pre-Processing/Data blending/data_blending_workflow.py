import json
import time
import sys
from data_blender import *

# inputs
gdf_pickle_path = sys.argv[1]; time.sleep(1); gdf = pickle.load(open(gdf_pickle_path, "rb"))
start = sys.argv[2]
end = sys.argv[3]
resolution = float(sys.argv[4])
country = sys.argv[5]
bbox_size = float(sys.argv[6])
ancillary_ext = sys.argv[7]
lidar_ext = sys.argv[8]
multispectral_ext = sys.argv[9]
narrow_swath_ext = sys.argv[10]
downloaded_files_path = sys.argv[11]
processed_files_folder = sys.argv[12]

extensions = {"ancillary": ancillary_ext,
              "narrow-swath": narrow_swath_ext,
              "lidar": lidar_ext,
              "multispectral": multispectral_ext}

# parameters/metadata
with open("workflow_params.json", "r", encoding="utf-8") as file:
    params = json.load(file)

paths = [params["lidar_path"],
         params['ecostress_path'],
         params['sentinel2_path']
         ]

scripts = [params["lidar_script"],
           params['ecostress_script'],
           params['sentinel2_script']
           ]

name = get_name(gdf, start, end)
dataset_paths, merged_dataset_path = get_paths(name, processed_files_folder, extensions)

temporary_gdf_pickle = f"{processed_files_folder}.gdf.pkl"

if not os.path.exists(dataset_paths[1]): # ECOSTRESS data
    print("Please enter your credentials to download ECOSTRESS data (visit https://appeears.earthdatacloud.nasa.gov/api/login for more info)")
    print("USERNAME:")
    username = input("")
    print("PASSWORD:")
    password = input("")
else:
    username = ""
    password = ""

args = [
    [temporary_gdf_pickle, name, str(resolution), processed_files_folder, extensions["lidar"], country], # LIDAR
    [temporary_gdf_pickle, start, end, name, processed_files_folder, str(bbox_size), extensions["narrow-swath"], username, password, downloaded_files_path], # ECOSTRESS
    [temporary_gdf_pickle, start, end, name, processed_files_folder, str(bbox_size), extensions["multispectral"], downloaded_files_path] # SENTINEL 2 MSI
]

pickle.dump(gdf, open(temporary_gdf_pickle, "wb"))  # temporary pickle to transmit to subscripts

execute(args, dataset_paths, paths, scripts)

os.remove(temporary_gdf_pickle) # remove temporary pickle

print("Executing data blending workflow...")
time.sleep(1)

ancillary_ds, narrow_swath_ds = merge(dataset_paths, gdf)
pickle.dump(ancillary_ds, open(merged_dataset_path, "wb"))
print(f'File registered as {merged_dataset_path}.')

# update LST dataset
pickle.dump(narrow_swath_ds, open(dataset_paths[1], "wb"))

print("--> Data blending workflow is done.")