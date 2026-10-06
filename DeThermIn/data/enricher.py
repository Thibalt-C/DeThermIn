import numpy as np
from cmcrameri.cm import cmaps
import xarray as xr
import pvlib
from scipy.ndimage import distance_transform_edt
import matplotlib.colors as colors
import pandas as pd
import datetime
from tqdm import tqdm
from typing import Optional


def get_sun_loc(dataset : xr.Dataset) -> pvlib.location.Location :
    """
    Aims to obtain the sun location for a given dataset.
    This includes elevation and azimuth angles, and can eventually be used for further Sun-related computations.
    """

    wgs84_coords = dataset.DSM.rio.reproject(dst_crs="EPSG:4326").coords

    loc = pvlib.location.Location(latitude=wgs84_coords["y"].values.mean(),
                                  longitude=wgs84_coords["x"].values.mean(),
                                  altitude=dataset.DSM.mean().values)

    return loc


def get_sun_pos(
        dataset : xr.Dataset,
        loc : pvlib.location.Location
) -> xr.Dataset :

    sun_pos = loc.get_solarposition(dataset.time.values)

    for angle in ["azimuth", "elevation"]:
        dataset[angle] = xr.DataArray(data=sun_pos[angle].values,
                                      dims=["time"], coords={"time": dataset.time}
                                      )

    return dataset


def rebuild_shadow(
        dataset : xr.Dataset,
        verbose : Optional[bool] = True
) -> xr.Dataset :
    """
    Aims to retrieve the nearest corresponding shadow configuration for each considered timestamp.
    """

    if "time" in list(dataset.shadow.coords):
        print("shadow already rebuilt.")
        return dataset

    loc = get_sun_loc(dataset)

    dataset = get_sun_pos(dataset, loc)

    shadows = []

    iterator = tqdm(dataset.time.values,
                    total=len(dataset.time.values),
                    desc="Rebuilding shadows"
                    ) if verbose else dataset.time.values

    for t in iterator:

        e = dataset.elevation.sel(time=t)
        a = dataset.azimuth.sel(time=t)

        shadow = dataset.sel(time=t).sel(sun_elevation=e, method="nearest"
                                         ).sel(sun_azimuth=a, method="nearest"
                                               ).shadow

        if e<=0: # nighttime
            shadow = xr.DataArray(data=np.full_like(shadow, True), dims=["y", "x"], coords=shadow.coords)

        shadows.append(shadow)

    shadows = xr.concat(shadows, dim=("time"))

    dataset["shadow"] = xr.DataArray(shadows.data,
                                     dims=["time", "y", "x"],
                                     coords={"y": dataset.y, "x": dataset.x, "time": dataset.time},
                                     attrs={"standard_name": "Shadow",
                                            "units": "no units",
                                            "plot_kwargs": {"cmap": "grey_r", "vmin": 0, "vmax": 1, "add_colorbar": False}
                                            }
                                     )

    return dataset


def compute_irradiance(
        dataset : xr.Dataset,
        shadow : xr.DataArray,
        loc : pvlib.location.Location,
        time_axis : pd.DatetimeIndex,
) -> np.ndarray :

    sun_pos = loc.get_solarposition(time_axis)
    cs_irradiance_df = loc.get_clearsky(time_axis)

    nt = len(time_axis); ny = len(dataset.y); nx = len(dataset.x)

    slope_grad_expanded = np.broadcast_to(
        np.degrees(dataset.slope_grad.values).astype(np.float32)[None, ...],
        (nt, ny, nx)
    )

    aspect_expanded = np.broadcast_to(
        np.degrees(dataset.aspect.values).astype(np.float32)[None, ...],
        (nt, ny, nx)
    )

    elevation_expanded = np.broadcast_to(
        (90.0 - sun_pos["elevation"].values).astype(np.float32)[..., None, None],
        (nt, ny, nx)
    )

    azimuth_expanded = np.broadcast_to(
        sun_pos["azimuth"].values.astype(np.float32)[..., None, None],
        (nt, ny, nx)
    )

    dni_expanded = np.broadcast_to(
        cs_irradiance_df["dni"].values.astype(np.float32)[..., None, None],
        (nt, ny, nx)
    )

    ghi_expanded = np.broadcast_to(
        cs_irradiance_df["ghi"].values.astype(np.float32)[..., None, None],
        (nt, ny, nx)
    )

    dhi_expanded = np.broadcast_to(
        cs_irradiance_df["dhi"].values.astype(np.float32)[..., None, None],
        (nt, ny, nx)
    )

    irrad = pvlib.irradiance.get_total_irradiance(slope_grad_expanded,
                                                  aspect_expanded,
                                                  elevation_expanded,
                                                  azimuth_expanded,
                                                  dni_expanded,
                                                  ghi_expanded,
                                                  dhi_expanded
                                                  )

    if shadow.dtype != int:
        shadow = shadow.astype(int)
    shadow = abs(shadow.values - 1)  # 0 if shadowed else 1

    # change global irradiance to include the impact of shadowing
    irrad["poa_global"] = irrad["poa_direct"].astype(np.float32) * shadow + irrad["poa_diffuse"].astype(np.float32)

    return irrad["poa_global"]


def get_irradiance(
        dataset : xr.Dataset
) -> xr.Dataset :
    """
    Aims to compute the total irradiance per pixel for each timestamp.
    """

    loc = get_sun_loc(dataset)

    irrad = compute_irradiance(dataset, dataset.shadow, loc, pd.to_datetime(dataset.time.values))

    dataset["irradiance"] = xr.DataArray(data=irrad,
                                          dims=["time", "y", "x"],
                                          coords=[dataset.time, dataset.y, dataset.x],
                                          attrs={"standard_name": "Irradiance",
                                                 "units": "W/m^2",
                                                 "Description": "Computed irradiance using an isotropic model from pvlib library.",
                                                 "plot_kwargs": {"cmap": "jet", "vmin": 0}
                                                 })

    return dataset


def get_irradiated_energy(
        dataset : xr.Dataset,
        time_window : float, # hours
        frequency : float, # hours
        verbose : Optional[bool] = True
) -> xr.Dataset :

    loc = get_sun_loc(dataset)

    dataset["energy"] = xr.DataArray(np.full((len(dataset.time), len(dataset.y), len(dataset.x)), np.nan),
                                     dims=["time", "y", "x"],
                                     coords={"time": dataset.time, "y": dataset.y, "x": dataset.x},
                                     attrs={"standard_name": "Irradiated energy",
                                            "units": "MJ/m^2",
                                            "description": f"Irradiated energy for clear sky conditions in a time window of {time_window} hours, computed using a {frequency}-hour frequency",
                                            "plot_kwargs": {"cmap": "jet"}
                                            })

    iterator = tqdm(enumerate(dataset.time), total=len(dataset.time), desc=f"Computing clear sky irradiated energy (last {time_window} hours)") if verbose else enumerate(dataset.time)

    for n,t in iterator:

        timestamp = pd.Timestamp(t.values).to_pydatetime()


        time_axis = [timestamp -  datetime.timedelta(seconds=time_delta) for time_delta in range(0, int(time_window*60*60), int(frequency*60*60))]
        time_axis = pd.DatetimeIndex(time_axis)

        sun_pos = loc.get_solarposition(time_axis)

        elevation = sun_pos["elevation"].values
        azimuth = sun_pos["azimuth"].values

        shadows = []

        for i in range(len(time_axis)):

            shadow = dataset.sel(sun_elevation=elevation[i], method="nearest").sel(sun_azimuth=azimuth[i], method="nearest").shadow
            if "subtime" not in list(shadow.coords):
                shadow = shadow.assign_coords({"subtime": np.datetime64(time_axis[i], "ns")})
            shadows.append(shadow)

        shadows = xr.concat(shadows, dim=("subtime"))

        irrad = compute_irradiance(dataset, shadows, loc, time_axis)

        energies = irrad * (frequency * 3600) / 1e6 # to MJ/m^2
        dataset["energy"].data[n,:,:] = xr.DataArray(energies,
                                                     dims=["subtime", "y", "x"]
                                                     ).sum("subtime")
        del energies, shadows, irrad

    return dataset


def get_illumination(dataset, frequency=0.25, verbose=True):

    loc = get_sun_loc(dataset)

    date = pd.to_datetime(dataset.time.values[0])
    init_datetime = datetime.datetime(date.year, date.month, date.day, 0, 0)

    time_axis = [init_datetime + datetime.timedelta(seconds=time_delta)
                 for time_delta in range(0, int(24*60*60), int(frequency*60*60))]

    cs_irradiance_df = loc.get_clearsky(pd.DatetimeIndex(time_axis))
    sun_pos = loc.get_solarposition(pd.DatetimeIndex(time_axis))
    cs_irradiance_df = cs_irradiance_df.where(cs_irradiance_df!=0, axis=0).dropna() # only day-time
    df = pd.concat([sun_pos, cs_irradiance_df], axis=1).dropna()
    time_axis = df.index.values

    elevation = df["elevation"].values
    azimuth = df["azimuth"].values

    shadows = []

    iterator = tqdm(range(len(time_axis)), desc=f"Computing daily illumination") if verbose else range(len(time_axis))

    for t in iterator:
        shadow = dataset.sel(sun_elevation=elevation[t], method="nearest").sel(sun_azimuth=azimuth[t], method="nearest").shadow
        shadows.append(shadow)
    shadows = xr.concat(shadows, dim=("subtime"))

    illumination = abs(shadows.sum("subtime") - len(time_axis)) * frequency # in hours of illumination

    dataset["illumination"] = xr.DataArray(illumination, dims=['y', 'x'],
                                coords={"y": dataset.y, "x": dataset.x},
                                attrs={"standard_name": "Daily illumination", "units": "h/day",
                                       "plot_kwargs":{"cmap": cmaps["grayC"], "vmin":0}}
                                ).astype('float32')

    return dataset


def distance_grid(dataset, feature, true_value, type_grid=None):

    if type_grid == "not_equal":
        mask = dataset[feature].values != true_value
    else:
        mask = dataset[feature].values <= true_value

    distance = distance_transform_edt(mask)
    distance[dataset.mask.values!=1] = np.nan

    return distance


def get_distance_grids(dataset, vegetation_threshold=5):

    distance_to_vegetation = distance_grid(dataset, "vegetation", vegetation_threshold)
    distance_to_water = distance_grid(dataset, "water_mask", 1, "not_equal")

    dataset["cold_spot_distance"] = xr.DataArray(np.min(np.stack([distance_to_vegetation, distance_to_water], axis=-1), axis=2),
                                                 dims=("y","x"),
                                                 attrs={"standard_name": "Distance to water or vegetation", "units": "m",
                                                        "description": f"Uses vegetation>{vegetation_threshold} to determine vegetation distance and water mask to determine water distance.",
                                                        "plot_kwargs": {"cmap": "grey"}})

    return dataset


def get_hsv_channels(dataset):

    rgb_array = dataset.orthoimage.values

    if rgb_array.shape[-1]!=3:
        rgb_array = np.transpose(rgb_array, (1, 2, 0))

    rgb_normalized = rgb_array / 255.0
    hsv_array = colors.rgb_to_hsv(rgb_normalized)

    dataset["hue"] = xr.DataArray(hsv_array[:,:,0], dims=["y", "x"], attrs={"standard_name": "Hue",
                                                                     "plot_kwargs": {"cmap": "hsv"}})

    dataset["saturation"] = xr.DataArray(hsv_array[:,:,1], dims=["y", "x"], attrs={"standard_name": "Saturation",
                                                                            "plot_kwargs": {"cmap": cmaps["bilbao"]}})

    dataset["brightness"] = xr.DataArray(hsv_array[:,:,2], dims=["y", "x"], attrs={"standard_name": "Brightness",
                                                                            "plot_kwargs": {"cmap": cmaps["grayC"]}})

    return dataset



def enrich_dataset(
        dataset : xr.Dataset,
        verbose : Optional[bool] = True
) -> xr.Dataset :

    if verbose:
        print("Getting extra layers (HSV channels, distance grids)...")
    dataset = get_distance_grids(dataset)
    dataset = get_hsv_channels(dataset)

    dataset = get_illumination(dataset, verbose=verbose)

    dataset = get_irradiated_energy(dataset, time_window=4, frequency=0.25, verbose=verbose)

    dataset = rebuild_shadow(dataset, verbose=verbose)

    if verbose:
        print("Computing irradiance...")
    dataset = get_irradiance(dataset)

    return dataset

