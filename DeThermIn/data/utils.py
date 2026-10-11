import numpy as np
# noinspection PyUnusedImports
import rioxarray


def find_spatial_resolution(dataset): # measured in the easting direction only

    spatial_resolution = np.abs(dataset.x[0].values-dataset.x[1].values)

    if dataset.rio.crs.to_epsg() == 4326:
        spatial_resolution = spatial_resolution * 111320

    spatial_resolution = np.round(spatial_resolution, 2)

    return spatial_resolution


def find_temporal_resolution(dataset):

    if "time" in dataset.dims:
        timedelta_ns = (dataset.time.max().values - dataset.time.min().values) / len(dataset.time)
        temporal_resolution = int(timedelta_ns / 1e9) / 60 / 60
    else:
        temporal_resolution = None

    return temporal_resolution


def find_status(filename):

    status = filename.split(".")[0].split("_")[-1]
    try:  # change for blended status
        status = int(status)
        status = ""

    except:
        pass

    return status


def find_name(filename, status):

    if status == "":
        name = filename.split(".pkl")[0].split("/")[-1]
    else:
        name = filename.split(".pkl")[0].split(status)[0].split("/")[-1]
        name = name[:-1]

    return name


def dataset_compressor(dataset, name=None, verbose=True):

    original_size = dataset.nbytes / 1e9 # GB

    for layer in dataset.keys():
        dtype = dataset[layer].dtype
        if dtype == float:
            dataset[layer] = dataset[layer].astype("float32")
        elif (dtype == int) & (dtype != np.uint8):
            dataset[layer] = dataset[layer].astype("uint8")

        unique = np.unique(dataset[layer].values)
        valid_mask = ~np.isnan(dataset[layer].values)
        if (len(unique[np.isfinite(unique)]) <= 2):
            dataset[layer] = dataset[layer].astype("bool")
            dataset[layer] = dataset[layer].where(valid_mask, False)

    for coord in list(dataset.coords):
        dtype = dataset.coords[coord].dtype
        if dtype == float:
            dataset.coords[coord] = dataset.coords[coord].astype("float32")

    compressed_size = dataset.nbytes / 1e9

    if verbose:
        if name is not None:
            print(f"{original_size:.2f} GB -> {compressed_size:.2f} GB ({name})")
        else:
            print(f"{original_size:.2f} GB -> {compressed_size:.2f} GB")

    return dataset