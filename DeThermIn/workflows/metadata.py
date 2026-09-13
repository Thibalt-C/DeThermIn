"""
This module is loading the metadata of the project that can be edited in the info.json file.
"""

import geopandas as gpd
import json
import datetime

with open("DeThermIn/workflows/info.json", "rb") as f:
    metadata = json.load(f)

available_areas = gpd.read_file(
    metadata["geofeatures_path"]+metadata["available_areas_file"]
)

resolution = metadata["resolution"]
start = datetime.datetime.strptime(metadata["start_date"], "%Y-%m-%d")
end = datetime.datetime.strptime(metadata["end_date"], "%Y-%m-%d")