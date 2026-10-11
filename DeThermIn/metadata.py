"""
This module is loading the metadata of the project that can be edited in the info.json and parameters.json files.
"""

import geopandas as gpd
import json
import datetime
import os


# paths to the metadata files
json_filepath_metadata = "DeThermIn/variables/info.json"
json_filepath_parameters = "DeThermIn/variables/parameters.json"
json_filepath_parameters_description = "DeThermIn/variables/parameters_description.json"

with open(json_filepath_metadata, "rb") as f:
    metadata = json.load(f)

with open(json_filepath_parameters, "rb") as f:
    parameters = json.load(f)

with open(json_filepath_parameters_description, "rb") as f:
    parameters_description = json.load(f)


# file containing the available areas
available_areas = gpd.read_file(
    metadata["geofeatures_path"]+metadata["available_areas_file"]
)

# pre processing parameters
resolution = parameters["resolution"]
start = datetime.datetime.strptime(parameters["start_date"], "%Y-%m-%d")
end = datetime.datetime.strptime(parameters["end_date"], "%Y-%m-%d")
bbox_size = int(parameters["larger_bbox_size_km"])

# features used for training
training_features_path = metadata["training_path"] + metadata["training_features_file"]
with open(training_features_path, "rb") as f:
    training_features = json.load(f)

# paths to the metadata files
lidar_path = metadata["lidar_metadata_path"]
data_blending_path = metadata["data_blending_metadata_path"]

# extensions of the processed files
extensions = metadata["extensions"]


# function to update the paths of downloaded LiDAR data
def update_lidar_metadata(path):

    with open(lidar_path, "r") as f:
        lidar_metadata = json.load(f)

    for key in lidar_metadata.keys():
        lidar_metadata[key] = os.path.join(path, f"LIDAR/{key}/")

    with open(lidar_path, "w") as f:
        json.dump(lidar_metadata, f, indent=4)

# repercussion of the global parameters
update_lidar_metadata(parameters["downloaded_data_path"])