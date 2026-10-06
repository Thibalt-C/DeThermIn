import laspy
import numpy as np
import requests
import os
import pandas as pd
import xarray as xr
from tqdm import tqdm
from .folder_manager import lidar_tiles



def download_lidar(lidar_urls,
                   lidar_filenames,
                   lidar_path=lidar_tiles,
                   decimation_factor=10,
                   verbose=True
                   ):

    clouds = []

    iterator = tqdm(enumerate(lidar_filenames), total=len(lidar_filenames), desc="Loading LiDAR data") if verbose else enumerate(lidar_filenames)

    for n, filename in iterator:

        if not os.path.exists(lidar_path+filename):
            with requests.get(lidar_urls[n],stream=True) as response:
                response.raise_for_status()
                with open(lidar_path+filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

        decimated_chunks = read_decimated_laz(lidar_path+filename, decimation_factor=decimation_factor)
        clouds.extend(decimated_chunks)

    return clouds


import numpy as np
import laspy


def read_decimated_laz(filepath,
                       decimation_factor=10,
                       chunk_size=1_000_000):

    selected_chunks = []

    with laspy.open(filepath) as f:

        for points in f.chunk_iterator(chunk_size):

            mask = np.arange(len(points)) % decimation_factor == 0

            selected_chunks.append(points[mask])

    return selected_chunks


def density_per_point(x,
                      y,
                      bounds,
                      lengths,
                      name
                      ):

    dx = abs(bounds[0]-bounds[2])/lengths[0]
    dy = abs(bounds[1]-bounds[3])/lengths[1]
    x_bins = np.linspace(bounds[0]-dx/2, bounds[2]+dx/2, lengths[0]+1)
    y_bins = np.linspace(bounds[1]-dy/2, bounds[3]+dy/2, lengths[1]+1)

    x_indices = np.digitize(x, x_bins) - 1
    y_indices = np.digitize(y, y_bins) - 1

    valid_mask = (
            (x_indices >= 0) &
            (x_indices < lengths[0]) &
            (y_indices >= 0) &
            (y_indices < lengths[1])
    )

    x_indices = x_indices[valid_mask]
    y_indices = y_indices[valid_mask]

    df = pd.DataFrame({
        'x_idx': x_indices,
        'y_idx': y_indices,
        'class': ['class']*len(x_indices)
    })

    counts = df.groupby(['x_idx', 'y_idx', 'class']).size().unstack(fill_value=0)

    density_map = np.zeros((len(y_bins)-1, len(x_bins)-1))

    iterator = tqdm(counts.iterrows(), total=counts.size, desc=f"Computing {name} density map") if name!="" else counts.iterrows()

    for (x, y), row in iterator:
        density_map[-y, x] = row.get('class',0)

    return density_map



def get_vegetation_cover(dataset, clouds, verbose=True):

    x = []
    y = []

    for cloud in clouds:
        vegetation = np.where(cloud.classification==1)
        x.extend(cloud.x[vegetation])
        y.extend(cloud.y[vegetation])

    bounds = [dataset.x.min().values, dataset.y.min().values,
                  dataset.x.max().values, dataset.y.max().values]
    lengths = [len(dataset.x), len(dataset.y)]
    vegetation = density_per_point(x, y, bounds, lengths, "vegetation cover" if verbose else "")

    dataset["vegetation"] = xr.DataArray(vegetation,
                                     dims=('y', 'x'),
                                     attrs={"standard_name": "Vegetation cover",
                                            "units": "pt/pixel",
                                            "description": "Actual class is 'unclassified', but 99% of it is assumed to be vegetation.",
                                            "plot_kwargs":{"cmap":"Greens",
                                                           "vmin":0,
                                                           "vmax":vegetation.max()
                                                           }
                                            }
                                     )

    return dataset