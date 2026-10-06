import numpy as np
import xarray as xr
from .metadata import *
from .folder_manager import *
from .tiles_tools import *
from .point_cloud_tools import *
from .utils import *


def download_data(gdf,
                  decimation_factor = 5,
                  lidar_decimation_factor = 10,
                  build_dataset=True,
                  data_for_derivation="DSM",
                  threshold_for_warning=10,
                  verbose=True
                  ):

    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    urls, filenames = get_metadata(gdf)
    projection = "EPSG:28992" # Amersfoort / RD New for mainland Netherlands

    if len(urls[-1]) >= threshold_for_warning:
        print(ImportWarning(f"The selected area contains {len(urls[-1])} LiDAR tiles. Please insure there is enough storage before proceeding."))
        _ = input("Press any key to continue or [Q] to exit...").strip().lower()
        if _ == 'q':
            raise KeyboardInterrupt()

    check_folders()

    sets = download_tiles(urls[1], urls[0], filenames[1], filenames[0],
                          decimation_factor=decimation_factor,
                          verbose=verbose)

    unbuilt_dataset = dem_dsm_xarray(sets, projection)
    unbuilt_dataset = clip_dataset(unbuilt_dataset, gdf)

    clouds = download_lidar(urls[2], filenames[2], decimation_factor=lidar_decimation_factor, verbose=verbose)

    if build_dataset:

        dataset = get_water_and_building_masks(unbuilt_dataset)

        dataset.DSM.values[np.isnan(dataset.DSM.values)
                                   & dataset.mask.values] = 0.0  # put water areas to 0 meters instead of NaN

        dataset = compute_subproducts(dataset,
                                      resolution=original_resolution*decimation_factor,
                                      data_for_derivation=data_for_derivation,
                                      verbose=verbose)
        dataset = get_vegetation_cover(dataset, clouds, verbose=verbose)
        dataset = clip_dataset(dataset, gdf)

        dataset = compress_dataset(dataset, verbose=verbose)

        return dataset

    else:
        return unbuilt_dataset, clouds



def get_water_and_building_masks(dataset):

    if "mask" in list(dataset.keys()):
        spatial_mask = dataset.mask==1
    else:
        spatial_mask = np.full_like(dataset.DSM.values, True)

    water_mask = np.isnan(dataset.DSM) & spatial_mask
    dem_mask = np.isnan(dataset.DEM)
    building_mask = (~water_mask) & (dem_mask) & spatial_mask

    dataset["water_mask"] = xr.DataArray(water_mask,
                                    dims=('y', 'x'),
                                    attrs={"standard_name": "Water pixels",
                                           "units": "no units",
                                           "plot_kwargs":{"cmap":"Blues",
                                                          "add_colorbar":False
                                                          }
                                           }
                                    )

    dataset["buildings_mask"] = xr.DataArray(building_mask,
                                        dims=('y', 'x'),
                                        attrs={"standard_name": "Building pixels",
                                               "units": "no units",
                                               "plot_kwargs": {"cmap": "Oranges",
                                                               "add_colorbar": False
                                                               }
                                               }
                                        )

    return dataset