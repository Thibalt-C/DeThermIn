import subprocess
import os
import datetime
from geopandas import GeoDataFrame
from ..data import read_datacube, DataCube
from ..metadata import metadata, extensions, parameters


def generate_ground_truth(
        resolution : float,
        time_extent : int
) -> DataCube :

    process = subprocess.Popen(["python", metadata["ground_truth_script"],
                                str(resolution),
                                parameters["processed_data_path"],
                                str(time_extent),
                                extensions["ground_truth"]
                                ],
                               cwd=os.path.join(os.path.curdir, metadata["ground_truth_path"]),
                               stdout=subprocess.PIPE,
                               universal_newlines=True,
                               bufsize=1,
                               env=os.environ.copy())

    lines = []
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        lines.append(line)

    process.wait()

    filepath = [line for line in lines if ".pkl" in line][-1].split(" ")[-1].split("\n")[0]

    dc = read_datacube(filepath)
    dc.compress(inplace=True)

    return dc


def download_geostationary_data(
        start : datetime.datetime,
        end : datetime.datetime,
        gdf : GeoDataFrame
) -> DataCube :

    gdf["code"] = "GroundTruth"

    filepath = parameters["geostationary_data_path"] + f"GroundTruth-{start.strftime("%Y%m%d")}-{end.strftime("%Y%m%d")}.pkl"

    if not os.path.exists(filepath):
        from ..scripts.pre_download_geostationary_data import predownload
        predownload(start, end, gdf)

    dc = read_datacube(filepath)
    dc.compress(inplace=True)
    dc.save()

    return dc


def train(training_datacube_path):

    process = subprocess.Popen(["python", metadata["training_script"],
                                training_datacube_path,
                                metadata["training_features_file"],
                                parameters["super_resolution_train_size"],
                                parameters["super_resolution_valid_size"],
                                parameters["super_resolution_max_epochs"],
                                parameters["super_resolution_batch_size"],
                                parameters["super_resolution_patch_size"],
                                parameters["super_resolution_learning_rate"],
                                parameters["super_resolution_lambda_ssim"]
                                ],
                               cwd=os.path.join(os.path.curdir, metadata["training_path"]),
                               stdout=subprocess.PIPE,
                               universal_newlines=True,
                               bufsize=1,
                               env=os.environ.copy())

    for line in iter(process.stdout.readline, ''):
        print(line, end='')

    process.wait()

    return