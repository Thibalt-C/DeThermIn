import numpy as np
import cmcrameri.cm as cm
import xarray as xr
import pvlib
from scipy.ndimage import distance_transform_edt
import matplotlib.colors as colors
import pandas as pd
import datetime
from tqdm import tqdm


def rebuild_shadow(dataset, verbose=True):

    if "time" in list(dataset.shadow.coords):
        print("shadow already rebuilt.")
        return dataset

    wgs84_coords = dataset.DSM.rio.reproject(dst_crs="EPSG:4326").coords

    lon, lat = wgs84_coords["x"].values.mean(), wgs84_coords["y"].values.mean()

    elevation = np.nanmean(dataset.DEM.values)

    loc = pvlib.location.Location(longitude=lon, latitude=lat, altitude=elevation)

    sun_pos = loc.get_solarposition(dataset.time.values)

    dataset["elevation"] = xr.DataArray(data=sun_pos["elevation"].values, dims=["time"], coords={"time": dataset.time})
    dataset["azimuth"] = xr.DataArray(data=sun_pos["azimuth"].values, dims=["time"], coords={"time": dataset.time})

    del sun_pos

    shadows = []

    iterator = tqdm(dataset.time.values, total=len(dataset.time.values), desc="Rebuilding shadows") if verbose else dataset.time.values

    for t in iterator:

        e = dataset.elevation.sel(time=t)
        a = dataset.azimuth.sel(time=t)
        #cloudy = dataset.cloudy_days.sel(time=t)

        shadow = dataset.sel(time=t).sel(sun_elevation=e, method="nearest").sel(sun_azimuth=a, method="nearest").shadow
        if e<=0: #| (cloudy==2):
            shadow = xr.DataArray(data=np.full_like(shadow, True), dims=["y", "x"], coords=shadow.coords)
        #if cloudy==1:
            #shadow = shadow.where(dataset.sel(time=t).cloud_mask!=1, True)
        shadows.append(shadow)

    shadows = xr.concat(shadows, dim=("time"))


    dataset["shadow"] = xr.DataArray(shadows.data, dims=["time", "y", "x"], coords={"y": dataset.y, "x": dataset.x, "time": dataset.time})

    del shadows

    return dataset



def get_cloudy_days(dataset, verbose=True):

    cloudy_days = []

    iterator = tqdm(range(len(dataset.time)), desc="Getting cloudy days") if verbose else range(len(dataset.time))

    for t in iterator:
        cloudyness = dataset.isel({"time":t}).cloud_mask
        mask = (dataset.isel({"time":t}).mask==1) & (cloudyness==1)

        if mask.any():
            if 0 not in np.unique(cloudyness.where(dataset.isel({"time":t}).mask==1).values):
                cloudy_days.append(2)
            else:
                cloudy_days.append(1)
        else:
            cloudy_days.append(0)

    dataset["cloudy_days"] = xr.DataArray(cloudy_days, dims=["time"], coords={"time": dataset.time})

    return dataset


def get_irradiance(dataset, verbose=True):

    wgs84_coords = dataset.DSM.rio.reproject(dst_crs="EPSG:4326").coords

    loc = pvlib.location.Location(latitude=wgs84_coords["y"].values.mean(),
                                  longitude=wgs84_coords["x"].values.mean(),
                                  altitude = dataset.DSM.mean().values)

    cs_irradiance_df = loc.get_clearsky(dataset.time.indexes["time"])

    del loc

    irrad = pvlib.irradiance.get_total_irradiance(
        np.tile(
            np.expand_dims(np.degrees(dataset.slope_grad.values), axis=-1),
            len(dataset.time)),
        np.tile(
            np.expand_dims(np.degrees(dataset.aspect.values), axis=-1),
            len(dataset.time)),
        np.tile(90 - dataset.elevation.values, (len(dataset.y),len(dataset.x),1)),
        np.tile(dataset.azimuth.values, (len(dataset.y),len(dataset.x),1)),
        np.tile(cs_irradiance_df["dni"].values, (len(dataset.y),len(dataset.x),1)),
        np.tile(cs_irradiance_df["ghi"].values, (len(dataset.y),len(dataset.x),1)),
        np.tile(cs_irradiance_df["dhi"].values, (len(dataset.y),len(dataset.x),1))
    )

    if dataset.shadow.dtype!=int:
        dataset["shadow"] = dataset.shadow.astype(int)
    shadow = abs(dataset.shadow.values - 1) # 1 if clear 0 if shadowed

    irrad["poa_global"] = irrad["poa_direct"] * np.transpose(shadow, (1,2,0)) + irrad["poa_diffuse"]

    irradiation = np.full_like(irrad["poa_global"], np.nan)

    del irrad["poa_direct"]
    del irrad["poa_diffuse"]
    del irrad["poa_sky_diffuse"]

    iterator = tqdm(range(irradiation.shape[2]), desc="Computing irradiance") if verbose else range(irradiation.shape[2])

    for t in iterator:
        # if dataset.cloudy_days.isel(time=t)==2:
        #     irradiation[:,:,t] = irrad["poa_sky_diffuse"][:,:,t]
        # elif dataset.cloudy_days.isel(time=t)==1:
        #     irradiation[:,:,t][dataset.isel(time=t).cloud_mask==0] = irrad["poa_global"][:,:,t][dataset.isel(time=t).cloud_mask==0]
        #     irradiation[:,:,t][dataset.isel(time=t).cloud_mask==1] = irrad["poa_diffuse"][:,:,t][dataset.isel(time=t).cloud_mask==1]
        # else:
        irradiation[:,:,t] = irrad["poa_global"][:,:,t]


    dataset["irradiance"] = xr.DataArray(data=irradiation,
                                          dims=["y", "x", "time"],
                                          coords=[dataset.y, dataset.x, dataset.time],
                                          attrs={"standard_name": "Irradiance",
                                                 "units": "W/m^2",
                                                 "Description": "Computed irradiance using an isotropic model from pvlib library.",
                                                 "plot_kwargs": {"cmap": "jet"}, "vmin": 0})

    del irradiation

    return dataset


def distance_grid(dataset, feature, true_value, type=None):

    if type == "not_equal":
        mask = dataset[feature].values != true_value
    else:
        mask = dataset[feature].values <= true_value

    distance = distance_transform_edt(mask)
    distance[dataset.mask.values!=1] = np.nan

    return distance


def get_distance_grids(dataset, vegetation_threshold=50):

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
                                                                            "plot_kwargs": {"cmap": cm.bilbao}})

    dataset["brightness"] = xr.DataArray(hsv_array[:,:,2], dims=["y", "x"], attrs={"standard_name": "Brightness",
                                                                            "plot_kwargs": {"cmap": cm.grayC}})

    return dataset



def get_irradiance_loading(dataset,
                           time_window=4, # hours
                           frequency= 0.25, # hours
                           verbose = True
                           ):
    wgs84_coords = dataset.DSM.rio.reproject(dst_crs="EPSG:4326").coords

    lon, lat = wgs84_coords["x"].values.mean(), wgs84_coords["y"].values.mean()

    elevation = np.nanmean(dataset.DEM.values)

    loc = pvlib.location.Location(longitude=lon, latitude=lat, altitude=elevation)

    energy_list = []

    aspect = dataset.aspect.values.copy()
    slope_grad = dataset.slope_grad.copy()

    iterator = tqdm(dataset.time, total=len(dataset.time), desc=f"Computing clear sky irradiated energy (last {time_window} hours)") if verbose else dataset.time

    for t in iterator:

        timestamp = pd.Timestamp(t.values).to_pydatetime()


        time_axis = [timestamp -  datetime.timedelta(seconds=time_delta) for time_delta in range(0, int(time_window*60*60), int(frequency*60*60))]

        dt = (time_axis[0] - time_axis[-1]).total_seconds()

        cs_irradiance_df = loc.get_clearsky(pd.DatetimeIndex(time_axis))
        sun_pos = loc.get_solarposition(time_axis)

        elevation = sun_pos["elevation"].values
        azimuth = sun_pos["azimuth"].values

        time_sample_dataset = dataset.sel(time=t)

        irrad = pvlib.irradiance.get_total_irradiance(
        np.tile(
            np.expand_dims(np.degrees(slope_grad), axis=-1),
            len(time_axis)
        ),
        np.tile(
            np.expand_dims(np.degrees(aspect), axis=-1),
            len(time_axis)
        ),
        np.tile(90 - elevation, (len(dataset.y),len(dataset.x),1)),
        np.tile(azimuth, (len(dataset.y),len(dataset.x),1)),
        np.tile(cs_irradiance_df["dni"].values, (len(dataset.y),len(dataset.x),1)),
        np.tile(cs_irradiance_df["ghi"].values, (len(dataset.y),len(dataset.x),1)),
        np.tile(cs_irradiance_df["dhi"].values, (len(dataset.y),len(dataset.x),1))
        )

        irradiances = []

        for i in range(len(time_axis)):

            shdw = abs(
                time_sample_dataset.sel(sun_elevation=elevation[i], method="nearest").sel(sun_azimuth=azimuth[i], method="nearest").shadow.values - 1
            ) # 1 if clear 0 if shadowed
            irradiance = shdw * irrad["poa_direct"][:,:,i] + irrad["poa_diffuse"][:,:,i]

            irradiance = irradiance * dt / 1e6 # to MJoules / m^2
            irradiance = xr.DataArray(irradiance, dims=["y", "x"], coords=[dataset.y, dataset.x])
            irradiance.coords["subtime"] = time_axis[i]

            irradiances.append(irradiance)

        energy = xr.concat(irradiances, dim=("subtime")).sum("subtime").rename("Irradiated energy")
        del irradiances

        energy_list.append(energy)
        del energy

    dataset["energy"] = xr.concat(energy_list, dim=("time"))
    del energy_list

    dataset.energy.attrs["standard_name"] = "Irradiated energy"
    dataset.energy.attrs["units"] = "MJ/m^2"
    dataset.energy.attrs["description"] = f"Irradiated energy for clear sky conditions in a time window of {time_window} hours, computed using a {frequency}-hour frequency"
    dataset.energy.attrs["plot_kwargs"] = {"cmap": "jet"}


    return dataset



def get_illumination(dataset, frequency=0.25, verbose=True):

    wgs84_coords = dataset.DSM.rio.reproject(dst_crs="EPSG:4326").coords

    loc = pvlib.location.Location(latitude=wgs84_coords["y"].values.mean(),
                                      longitude=wgs84_coords["x"].values.mean(),
                                      altitude = dataset.DSM.mean().values)

    date = pd.to_datetime(dataset.time.values[0])

    init_datetime = datetime.datetime(date.year, date.month, date.day, 0, 0)

    time_axis = [init_datetime + datetime.timedelta(seconds=time_delta)
                 for time_delta in range(0, int(24*60*60), int(frequency*60*60))]
    cs_irradiance_df = loc.get_clearsky(pd.DatetimeIndex(time_axis))
    sun_pos = loc.get_solarposition(time_axis)
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
                                       "plot_kwargs":{"cmap": cm.grayC, "vmin":0}}
                                ).astype('float32')

    return dataset