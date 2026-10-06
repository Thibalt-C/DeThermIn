from .DTC_model import get_relative_time, split_in_cycles, compute_LST
from ..metrics import rmse, normalised_rmse
from scipy.optimize import minimize
from tqdm import tqdm
import numpy as np
import xarray as xr
from typing import Optional, List, Tuple


def validation_DTC_fitting(
        ultra_coarse_dataset : xr.Dataset,
        alpha_monotonicity : List[float],
        alpha_zenith : List[float],
        sunset_shift : List[float]
) -> Tuple[List[dict], xr.Dataset, List[xr.Dataset]] :

    results = []

    for a1 in alpha_monotonicity:
        for a2 in alpha_zenith:
            for sun in sunset_shift:

                rebuilt, rebuilt_cycles = fit_DTC(ultra_coarse_dataset,
                                                  alpha_monotonicity=a1,
                                                  alpha_zenith=a2,
                                                  sunset_shift=sun
                                                  )
                RMSE = rmse(rebuilt.LST_rebuilt.values, rebuilt.LST.values)

                print(a1, a2, sun, RMSE)

                results.append({"alpha_monotonicity": a1, "alpha_zenith": a2, "sunset_shift": sun,
                                "RMSE": RMSE, "data": [rebuilt, rebuilt_cycles]})

    minRMSE = np.inf
    for result in results:

        RMSE = result["RMSE"]

        if RMSE < minRMSE:
            minRMSE = RMSE
            best_result = result

    rebuilt, rebuilt_cycles = best_result["data"]

    print(best_result["alpha_monotonicity"], best_result["alpha_zenith"], best_result["sunset_shift"])

    return results, rebuilt, rebuilt_cycles



def fit_DTC_params(
        cycle : xr.Dataset,
        alpha_1 : float,
        alpha_2 : float,
        sunset_shift : float,
        sunrise_LST : Optional[xr.DataArray] = None,
        mode : Optional[str] = "normal"
) -> xr.Dataset :

    # fixed parameter
    thermal_sunset = float(cycle.sunset_time) - sunset_shift

    if mode == "first": # for first cycle to begin at sunrise
        cycle = cycle.where(cycle.cycle_time > 0, drop=True)

    t = cycle.cycle_time.values # time relative to the sunrise

    # construction of the parameters grids
    for param in ["thermal_sunset",
                  "thermal_zenith",
                  "LST_max",
                  "LST_sunrise",
                  "LST_next_sunrise"]:
        cycle[param] = xr.DataArray(
            np.full(cycle.LST.rio.shape, np.nan),
            dims=("y", "x"),
            coords={"y": cycle.y, "x": cycle.x}
        )

    # pixel-wise regression
    for i in range(len(cycle.x)):
        for j in range(len(cycle.y)):

            pixel = cycle.isel(x=i, y=j)

            # separating morning, afternoon and nighttime
            daytimes = pixel.where(pixel.cycle_time < pixel.sunset_time, drop=True)
            morning = daytimes.where(daytimes.cycle_time < pixel.sunset_time/2, drop=True)
            afternoon = daytimes.where(daytimes.cycle_time >= pixel.sunset_time/2, drop=True)
            nighttimes = pixel.where(pixel.cycle_time >= pixel.sunset_time, drop=True)
            morning_finite = morning.LST.values[np.isfinite(morning.LST.values)]
            afternoon_finite = afternoon.LST.values[np.isfinite(afternoon.LST.values)]
            night_finite = nighttimes.LST.values[np.isfinite(nighttimes.LST.values)]

            # applying all-day clear sky condition
            if (len(morning_finite) >= 2) & (len(afternoon_finite) >= 2) & (len(night_finite) >= 2):

                # managing the boundary condition between 2 cycles
                replace_sunrise_LST = False # default
                if mode == "normal":
                    if np.isfinite(sunrise_LST.isel(y=j, x=i).values):
                        replace_sunrise_LST = True

                # initial guess
                thermal_zenith0 = float(pixel.sunset_time/2)
                lst_zen0 = float(pixel.LST.max())
                lst_next0 = float(pixel.LST.min())
                lst_sunrise0 = float(pixel.LST.min())

                if replace_sunrise_LST:
                    x0 = np.array([thermal_zenith0, lst_zen0, lst_next0])
                    lst_sunrise_fixed = float(sunrise_LST.isel(y=j, x=i))
                    bounds = ((thermal_zenith0, thermal_zenith0+1), (0, 100), (-20, 50))
                else:
                    x0 = np.array([thermal_zenith0, lst_zen0, lst_sunrise0, lst_next0])
                    bounds = ((thermal_zenith0, thermal_zenith0+1), (0, 100), (-20, 50), (-20, 50))

                def cost_func(params):

                    if replace_sunrise_LST:
                        thermal_zenith, lst_zen, lst_next = params.copy()
                        lst_sunrise = lst_sunrise_fixed
                    else:
                        thermal_zenith, lst_zen, lst_sunrise, lst_next = params.copy()

                    pred = compute_LST(t,
                                       thermal_sunset, thermal_zenith,
                                       lst_zen, lst_sunrise, lst_next)

                    true = pixel.LST.values
                    cost = normalised_rmse(pred, true)

                    # ensure monotonicity in the afternoon between daytime and nighttime
                    after_zenith = pred[t>thermal_zenith]
                    diffs = np.diff(after_zenith)
                    violations = diffs[diffs > 0]
                    penalty = np.sum(violations)
                    cost += alpha_1 * penalty

                    # regularisation penalising too low LST predictions compared to max observed LST
                    reg_max = np.nanmax((np.nanmax(true)-lst_zen, 0))
                    cost += alpha_2 * reg_max

                    return cost

                res = minimize(cost_func, x0, bounds=bounds)

                if replace_sunrise_LST:
                    thermal_zenith, lst_zen, lst_next = res.x
                    lst_sunrise = lst_sunrise_fixed
                else:
                    thermal_zenith, lst_zen, lst_sunrise, lst_next = res.x

                cycle["thermal_sunset"].data[j,i] = thermal_sunset
                cycle["thermal_zenith"].data[j,i] = thermal_zenith
                cycle["LST_max"].data[j,i] = lst_zen
                cycle["LST_sunrise"].data[j,i] = lst_sunrise
                cycle["LST_next_sunrise"].data[j,i] = lst_next

    LST_rebuilt = compute_LST(cycle.cycle_time.values,
                              cycle.thermal_sunset.values,
                              cycle.thermal_zenith.values,
                              cycle.LST_max.values,
                              cycle.LST_sunrise.values,
                              cycle.LST_next_sunrise.values,
                              mode = "3D_array"
                              )

    cycle["LST_rebuilt"] = xr.DataArray(LST_rebuilt,
                                        dims=("time", "y", "x"),
                                        coords={"y": cycle.y, "x": cycle.x,
                                                "time": cycle.time, "cycle_time": cycle.cycle_time},
                                        attrs = {"plot_kwargs": {"cmap": "RdYlBu_r"},
                                                 "standard_name": "DTC-derived Land Surface Temperature",
                                                 "units": "°C"}
                                        )

    return cycle


def fit_DTC(
        geostationary_dataset : xr.Dataset,
        alpha_monotonicity : float,
        alpha_zenith : float,
        sunset_shift : float
) -> Tuple[xr.Dataset, List[xr.Dataset]] :

    cycles = split_in_cycles(geostationary_dataset)

    keys = list(cycles.keys())

    # case of the first cycle: no previous cycle so no constrain on sunrise LST
    first_cycle = get_relative_time(cycles[keys[0]])
    first_cycle = fit_DTC_params(first_cycle,
                                 mode="first",
                                 alpha_1=alpha_monotonicity,
                                 alpha_2=alpha_zenith,
                                 sunset_shift=sunset_shift)

    # adding to the list of cycles and initialising the iteration
    rebuilt_cycles = [first_cycle]
    previous_cycle = first_cycle
    iterator = tqdm(keys[1:], desc="Fitting geostationary to DTC_model cycles")

    # iterating over the remaining cycles
    for key in iterator:
        cycle = get_relative_time(cycles[key])
        cycle = fit_DTC_params(cycle,
                               mode="normal",
                               sunrise_LST=previous_cycle.LST_next_sunrise.copy(deep=True),
                               alpha_1=alpha_monotonicity,
                               alpha_2=alpha_zenith,
                               sunset_shift=sunset_shift)
        rebuilt_cycles.append(cycle)
        previous_cycle = cycle

    rebuilt = xr.concat(rebuilt_cycles, dim="time")

    rebuilt = rebuilt.rio.write_crs(geostationary_dataset.rio.crs)

    return rebuilt