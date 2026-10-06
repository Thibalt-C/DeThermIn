"""
This module allows to launch the pre-processing of the data.
Two separated processes can be executed:
    - data blending (downloading all data except geostationary LST)
    - geostationary data extraction from pre-downloaded data
"""

import subprocess
import os
import pickle
from datetime import datetime
from geopandas import GeoDataFrame
from ..data import read_datacube, DataCube
from ..metadata import metadata, extensions, parameters, available_areas

def launch_data_blending(
        gdf : GeoDataFrame,
        start : datetime,
        end : datetime,
        resolution : float,
        bbox_size : float
) -> DataCube :

    gdf_pickle = ".gdf.pkl"
    depth  = len(os.path.split(metadata["data_blending_path"]))
    gdf_pickle_path =  metadata["geofeatures_path"] + gdf_pickle
    pickle.dump(gdf, open(gdf_pickle_path, "wb"))

    country = available_areas.loc[available_areas.geometry.intersects(gdf.union_all())].reset_index(drop=True).iloc[0]["country"]

    process = subprocess.Popen(["python", metadata["data_blending_script"],
                                depth * "../" + gdf_pickle_path,
                                start.strftime("%m-%d-%Y"),
                                end.strftime("%m-%d-%Y"),
                                str(resolution),
                                country,
                                bbox_size,
                                extensions["blended_ancillary_data"],
                                extensions["lidar"],
                                extensions["multispectral"],
                                extensions["narrow-swath"],
                                parameters["downloaded_data_path"],
                                parameters["processed_data_path"]
                                ],
                               cwd=os.path.join(os.path.curdir, metadata["data_blending_path"]),
                               stdout=subprocess.PIPE,
                               universal_newlines=True,
                               bufsize=1,
                               env=os.environ.copy())

    lines = []
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        lines.append(line)

    process.wait()

    if len(lines)==1: # meaning already done
        filepath = [line for line in lines if ".pkl" in line][-1].split("Please delete ")[-1].split(" if you want to reprocess.")[0]
    else:
        filepath = [line for line in lines if "File registered as " in line][-1].split("File registered as ")[-1].split(".\n")[0]

    os.remove(gdf_pickle_path)

    dc = read_datacube(filepath)
    dc.compress(inplace=True)
    dc.save()

    return dc


def find_geostationary_data(
        geostationary_dc : DataCube,
        narrow_swath_dc : DataCube
) -> DataCube :

    geostationary_dc = DataCube(
        data=geostationary_dc.data.rio.write_crs("EPSG:4326").rio.clip_box(*narrow_swath_dc.data.rio.bounds(),
                                                                           crs=narrow_swath_dc.data.rio.crs),
        filepath=narrow_swath_dc.filepath.replace(narrow_swath_dc.status, extensions["geostationary"]))
    geostationary_dc.compress(inplace=True)
    geostationary_dc.save()

    return geostationary_dc