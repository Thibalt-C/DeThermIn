import numpy as np
import xarray as xr
import datetime
import pandas as pd
import pvlib


def get_sunrise_sunset(dataset, time_precision = 1):
    """
    Aims to obtain the sunrise and sunset times for each day of the time series.
    """

    loc = pvlib.location.Location(
        longitude=dataset.x.mean().values,
        latitude=dataset.y.mean().values
    )

    sunsets = [] ; sunrises = []

    dataset = dataset.assign_coords(day=("time", dataset.time.dt.floor("D").data))

    for day in np.unique(dataset.day.values):

        extent = 24 * 60 / time_precision
        dt = datetime.timedelta(minutes=time_precision)
        timestamps = [pd.to_datetime(day) + m*dt
                      for m in range(0,int(extent)+time_precision)
                      ]

        sun_pos = loc.get_solarposition(timestamps)
        elevation = sun_pos["elevation"].to_numpy()

        sunset_not_done, sunrise_not_done = True, True

        for i in range(len(elevation)-1):
            if elevation[i] * elevation[i+1] < 0: # sunset or sunrise
                if (elevation[i] > 0) & sunset_not_done: # sunset
                    sunsets.append(timestamps[i]+0.5*dt)
                    sunset_not_done = False
                if (elevation[i] < 0) & sunrise_not_done: # sunrise
                    sunrises.append(timestamps[i]+0.5*dt)
                    sunrise_not_done = False

    dataset["sunrise"] = xr.DataArray(sunrises, coords={"day": np.unique(dataset.day)})
    dataset["sunset"] = xr.DataArray(sunsets, coords={"day": np.unique(dataset.day)})

    return dataset


def assign_cycle(dataset):
    """
    Assigns to each time step the corresponding diurnal cycle.
    """

    sunrises = dataset.sunrise.to_pandas().sort_values()
    times = dataset.time.to_pandas()

    idx = sunrises.searchsorted(times, side="right") - 1
    idx[idx < 0] = 0

    cycle_date = sunrises.iloc[idx].dt.floor("D").to_numpy()

    dataset["diurnal_cycle"] = xr.DataArray(
        cycle_date,
        coords={"time": dataset.time},
        dims="time"
    )

    return dataset


def split_in_cycles(dataset):
    """
    Splits the time series into diurnal cycles.
    """

    dataset = get_sunrise_sunset(dataset)
    dataset = assign_cycle(dataset)

    groups = {}

    for cid, g in dataset.groupby("diurnal_cycle"):

        sunrise = dataset.sunrise.sel(day=cid).item()
        sunset = dataset.sunset.sel(day=cid).item()

        g.attrs["sunrise"] = pd.to_datetime(sunrise)
        g.attrs["sunset"] = pd.to_datetime(sunset)

        g = g.drop_dims("day")

        groups[pd.to_datetime(cid).date()] = g

    return groups


def get_relative_time(cycle):
    """
    Converts absolute time of a diurnal cycle into relative time.
    """

    time_of_day = (pd.to_datetime(cycle.time.values) - cycle.attrs["sunrise"]
                   ).total_seconds() / 3600 # in hours

    cycle.coords["cycle_time"] = xr.DataArray(
        time_of_day,
        dims=("time"),
        coords={"time": cycle.time},
        attrs={"units": "hours"}
    )

    cycle["sunset_time"] = (cycle.attrs["sunset"] - cycle.attrs["sunrise"]).total_seconds() / 3600

    return cycle