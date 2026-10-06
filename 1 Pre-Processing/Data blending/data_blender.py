import subprocess
import os
import pickle
import xarray as xr
import rioxarray
from rasterio.enums import Resampling


def get_name(gdf, start, end):

    geometry = (gdf.to_crs("epsg:4326")).union_all()
    coords = list(geometry.centroid.coords)[0]
    coords_str = (f"{round(coords[0], 4)}_{round(coords[1], 4)}").replace('.','_')
    name = coords_str + '_' + start.replace('-', '_') + '_' + end.replace('-', '_')

    return name


def get_paths(name, pickle_files_folder, extensions):

    lidar_dataset_path = pickle_files_folder + name + f"_{extensions["lidar"]}.pkl"
    LST_dataset_path = pickle_files_folder + name + f"_{extensions["narrow-swath"]}.pkl"
    S2_dataset_path = pickle_files_folder + name + f"_{extensions["multispectral"]}.pkl"
    filepath = f"{pickle_files_folder}{name}_{extensions["ancillary"]}.pkl"

    dataset_paths = [lidar_dataset_path, LST_dataset_path, S2_dataset_path]

    return dataset_paths, filepath



def execute(args, dataset_paths, filepaths, scripts):

    lidar_args, ecostress_args, sentinel2_args = args
    lidar_dataset_path, LST_dataset_path, S2_dataset_path = dataset_paths
    lidar_path, ecostress_path, sentinel2_path = filepaths
    lidar_script, ecostress_script, sentinel2_script = scripts

    pass_lidar, pass_ecostress, pass_s2 = False, False, False

    if not os.path.exists(lidar_dataset_path):
        print("Downloading LiDAR data...")
        lidar_process = subprocess.Popen(
            ["python", lidar_script, *lidar_args],
            cwd=os.path.join(os.path.curdir, lidar_path), env=os.environ.copy())
    else:
        pass_lidar = True
        print("LiDAR data already processed.")

    if not os.path.exists(LST_dataset_path):
        print("Downloading narrow-swath LST...")
        ecostress_process = subprocess.Popen(
            ["python", ecostress_script, *ecostress_args],
            cwd=os.path.join(os.curdir, ecostress_path), env=os.environ.copy())
    else:
        pass_ecostress = True
        print("narrow-swath LST already processed.")

    if not os.path.exists(S2_dataset_path):
        print("Downloading multispectral data...")
        S2_process = subprocess.Popen(
            ["python", sentinel2_script, *sentinel2_args],
            cwd=os.path.join(os.curdir, sentinel2_path), env=os.environ.copy())
    else:
        pass_s2 = True
        print("multispectral data already processed.")

    if not pass_lidar:
        lidar_process.wait()
    if not pass_ecostress:
        ecostress_process.wait()
    if not pass_s2:
        S2_process.wait()

    return



def merge(dataset_paths, gdf):

    lidar_dataset_path, ECOSTRESS_dataset_path, S2_dataset_path  = dataset_paths

    # VHSR+HSR BLENDED
    with open(lidar_dataset_path, "rb") as f:
        lidar_dataset = pickle.load(f)
    with open(S2_dataset_path, "rb") as f:
        S2_dataset = pickle.load(f)
    S2_dataset = S2_dataset.rio.set_spatial_dims(x_dim="x", y_dim="y")
    S2_dataset = S2_dataset.rio.reproject_match(lidar_dataset)
    S2_dataset = S2_dataset.rio.clip(gdf.geometry.values, gdf.crs)
    ANCILLARY_ds = xr.merge([S2_dataset, lidar_dataset])

    # narrow-swath BLENDED
    with open(ECOSTRESS_dataset_path, "rb") as f:
        ECOSTRESS_dataset = pickle.load(f)
    with open(S2_dataset_path, "rb") as f:
        S2_dataset = pickle.load(f)
    S2_dataset = S2_dataset.rio.set_spatial_dims(x_dim="x", y_dim="y")
    S2_dataset = S2_dataset.rio.reproject_match(ECOSTRESS_dataset, resampling=Resampling.average)
    S2_dataset = S2_dataset.drop_vars(["red", "green", "blue", "NIR", "SWIR1", "SWIR2"])
    ECOSTRESS_dataset = xr.merge([S2_dataset, ECOSTRESS_dataset], compat="override")

    return ANCILLARY_ds, ECOSTRESS_dataset
