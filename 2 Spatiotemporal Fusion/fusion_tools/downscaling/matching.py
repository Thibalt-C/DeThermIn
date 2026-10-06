from ..DTC import split_in_cycles, get_relative_time, compute_LST
import xarray as xr
from rasterio.enums import Resampling
import numpy as np

def match_cycles(coarse_dataset, fine_dataset):

    matched = []

    fine_cycles = split_in_cycles(fine_dataset)

    coarse_bounds = coarse_dataset.rio.bounds()
    fine_bounds = fine_dataset.rio.bounds()

    dx_fine = abs(fine_bounds[2] - fine_bounds[0])
    dx_coarse = abs(coarse_bounds[2] - coarse_bounds[0])

    for key, cycle in fine_cycles.items():
        cycle = get_relative_time(cycle)

        t = cycle.cycle_time.values

        keep = xr.DataArray(coarse_dataset.diurnal_cycle.values==cycle.diurnal_cycle.values[0],
                            dims=("time"),
                            coords={"time": coarse_dataset.time})

        coarse_cycle = coarse_dataset.where(keep, drop=True)

        params = coarse_cycle.isel(time=0)


        coarse_LST = compute_LST(t, params.thermal_sunset.values, params.thermal_zenith.values,
                                 params.LST_max.values, params.LST_sunrise.values, params.LST_next_sunrise.values,
                                 mode = "3D_array"
                                 )

        coarse_LST = xr.DataArray(coarse_LST,
                                  dims=("time", "y", "x"),
                                  coords={"y": coarse_cycle.y, "x": coarse_cycle.x,
                                          "time": cycle.time}
                                  ).rio.write_crs(coarse_dataset.rio.crs)

        if dx_fine > dx_coarse:
            cycle = cycle.rio.clip_box(*coarse_bounds, crs=coarse_dataset.rio.crs)
        else:
            coarse_LST = coarse_LST.rio.clip_box(*fine_bounds, crs=fine_dataset.rio.crs)
        cycle["coarse_LST"] = coarse_LST.rio.reproject_match(cycle, resampling=Resampling.bilinear)
        cycle = cycle.where(np.isfinite(cycle.coarse_LST).any(axis=0), drop=True)
        cycle.coarse_LST.attrs = {"plot_kwargs": {"cmap": "RdYlBu_r"},
                                  "standard_name": "DTC-model derived Land Surface Temperature",
                                  "units": "°C"}

        matched.append(cycle)

    matched = xr.concat(matched, dim="time")

    return matched


def get_pairs(matched):

    # accounts for sea areas that are not measured for the geostationary LST
    always_nan = matched.coarse_LST.isnull().all("time").values

    mask = (
            ((always_nan)[None, :, :] | np.isfinite(matched.coarse_LST.values))
            & ((always_nan)[None, :, :] | np.isfinite(matched.LST.values))
    )
    mask = xr.DataArray(mask.all(axis=(1, 2)), dims=("time"), coords={"time": matched.time})

    pairs = matched.where(mask, drop=True)

    return pairs


def get_target(coarse_dataset, fine_dataset, day):

    diurnal_cycle = day.date()

    target = coarse_dataset.where(
        coarse_dataset.diurnal_cycle.dt.date == diurnal_cycle,
        drop=True
    )

    coarse_bounds = coarse_dataset.rio.bounds()
    fine_bounds = fine_dataset.rio.bounds()

    dx_fine = abs(fine_bounds[2] - fine_bounds[0])
    dx_coarse = abs(coarse_bounds[2] - coarse_bounds[0])

    if dx_fine > dx_coarse:
        target = target.rio.clip_box(*coarse_bounds, crs=coarse_dataset.rio.crs)
    else:
        target = target.rio.clip_box(*fine_bounds, crs=fine_dataset.rio.crs)

    target = target.rio.reproject_match(fine_dataset, resampling=Resampling.bilinear)
    target = target.where(np.isfinite(target.LST).any(axis=0), drop=True)


    return target