import numpy as np
import xarray as xr
from .DTC_params import get_LST_params, get_extra_time_params


def DTC_equation(t, t_max, ts, T0, Ta, omega, delta_T, k, mode="float"):
    """
    DTC_model equation.
    """
    if mode == "float":

        LST = np.zeros((len(t)))

        for i in range(len(t)):
            if t[i] < ts:
                LST[i] = T0 + Ta * np.cos((np.pi / omega) * (t[i] - t_max))
            else:
                A = T0 + delta_T
                B = Ta * np.cos((np.pi / omega) * (ts - t_max)) - delta_T
                C = k / (k + (t[i] - ts))
                LST[i] = A + (B * C)

    elif mode == "3D_array":

        LST_day = T0 + Ta * np.cos((np.pi / omega) * (t - t_max))

        A = T0 + delta_T
        B = Ta * np.cos((np.pi / omega) * (ts - t_max)) - delta_T
        C = k / (k + (t - ts))

        LST_night = A + (B * C)
        LST = np.where(t < ts, LST_day, LST_night)

    return LST


def compute_LST(t, thermal_sunset, thermal_zenith, thermal_zenith_LST, sunrise_LST, next_sunrise_LST, mode="float"):

    omega, u = get_extra_time_params(thermal_sunset, thermal_zenith)
    T0, Ta, omega, delta_T, k = get_LST_params(thermal_zenith_LST, sunrise_LST, next_sunrise_LST, thermal_sunset, omega, u)

    if mode == "3D_array":
        H, W = thermal_sunset.shape
        t = np.tile(t, (H, W, 1)).transpose(2, 0, 1)
        omega = np.tile(omega, (len(t), 1, 1))
        T0 = np.tile(T0, (len(t), 1, 1))
        Ta = np.tile(Ta, (len(t), 1, 1))
        delta_T = np.tile(delta_T, (len(t), 1, 1))
        k = np.tile(k, (len(t), 1, 1))
        thermal_sunset = np.tile(thermal_sunset, (len(t), 1, 1))
        thermal_zenith = np.tile(thermal_zenith, (len(t), 1, 1))

    params = [t, thermal_zenith, thermal_sunset, T0, Ta, omega, delta_T, k]

    LST = DTC_equation(*params, mode=mode)

    return LST


