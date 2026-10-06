import numpy as np

def get_extra_time_params(thermal_sunset, thermal_zenith):
    """
    Additional time parameters used for the DTC_model equation.
    """

    omega = (4 / 3) * thermal_zenith
    u = (np.pi / omega) * (thermal_sunset - thermal_zenith)
    return omega, u


def get_LST_params(thermal_zenith_LST, sunrise_LST, next_sunrise_LST, thermal_sunset, omega, u):
    """
    LST parameters used for the DTC_model equation.
    """

    Ta = (thermal_zenith_LST - sunrise_LST) / (np.cos(np.pi / 4) + 1)
    T0 = sunrise_LST + (Ta * np.cos(np.pi / 4))

    ts = thermal_sunset
    Tsr1 = next_sunrise_LST

    num = Ta * ((np.cos(u) * ((Ta * np.cos(u)) + T0 - Tsr1)) + ((np.pi / omega) * np.sin(u) * (24 - ts) * (T0 - Tsr1)))
    denom = T0 - Tsr1 - (Ta * (((np.pi / omega) * np.sin(u) * (24 - ts)) - np.cos(u)))
    delta_T = num / denom

    k = ( (Ta * np.cos(u)) - delta_T ) / ( ((Ta * np.pi) / omega) * np.sin(u) )

    return T0, Ta, omega, delta_T, k