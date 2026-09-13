"""
This module allows to launch the pre-processing of the data.
"""

import subprocess
import os
import pickle
from DeThermIn.data import read_datacube
from DeThermIn.workflows.metadata import metadata
from DeThermIn.workflows.metadata import available_areas


def launch(gdf, start, end, resolution):

    gdf_pickle = ".gdf.pkl"
    pickle.dump(gdf, open(metadata["geofeatures_path"] + gdf_pickle, "wb"))

    country = available_areas.where(gdf.geometry.intersects(available_areas.union_all())).dropna().reset_index(drop=True).iloc[0]["country"]

    start = start.strftime("%m-%d-%Y")
    end = end.strftime("%m-%d-%Y")

    process = subprocess.Popen(["python", metadata["data_blending_script"],
                                gdf_pickle, start, end, str(resolution), country],
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

    os.remove(metadata["geofeatures_path"] + gdf_pickle)

    dc = read_datacube(filepath)
    dc.compress(inplace=True)
    dc.save()

    return dc