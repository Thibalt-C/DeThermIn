import subprocess
import os
import copy
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime
from ..data import read_datacube, DataCube
from ..metadata import metadata, parameters, extensions


def launch_DTC_fitting(
        geostationary_path : str
) -> DataCube :
    """
    Launches the DTC fitting workflow and retrieves the produced DTC-fitted data.
    """

    process = subprocess.Popen(["python", metadata["DTC_fitting_script"],
                                geostationary_path,
                                parameters["DTC_alpha_1"],
                                parameters["DTC_alpha_2"],
                                parameters["DTC_sunset_shift"],
                                extensions["geostationary"],
                                extensions["DTC-fitted"]
                                ],
                               cwd=os.path.join(os.path.curdir, metadata["DTC_fitting_path"]),
                               stdout=subprocess.PIPE,
                               universal_newlines=True,
                               bufsize=1,
                               env=os.environ.copy())

    lines = []
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        lines.append(line)

    process.wait()

    filepath = geostationary_path.replace(extensions["geostationary"], extensions["DTC-fitted"])

    dc = read_datacube(filepath)

    dc.compress(inplace=True)

    dc.save()

    return dc



def launch_unmixing(
        narrow_swath_filepath : str,
        dtc_fitted_filepath : str,
        day : datetime = None
) -> DataCube :

    if day is None:
        available_days = get_available_days(read_datacube(dtc_fitted_filepath))
        for n, day in enumerate(available_days):
            print(f"[{n}] {day.strftime('%Y-%m-%d')}")
        n = input("Please select a day: ")
        day = available_days[int(n)]

    process = subprocess.Popen(["python", metadata["fusion_script"],
                                narrow_swath_filepath,
                                dtc_fitted_filepath,
                                day.strftime("%Y-%m-%d"),
                                extensions["narrow-swath"],
                                extensions["fusioned_LSTs"],
                                parameters["fusion_window_size"],
                                parameters["fusion_alpha_time"],
                                parameters["fusion_alpha_space"],
                                parameters["fusion_k"]
                                ],
                               cwd=os.path.join(os.path.curdir, metadata["fusion_path"]),
                               stdout=subprocess.PIPE,
                               universal_newlines=True,
                               bufsize=1,
                               env=os.environ.copy())

    lines = []
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        lines.append(line)

    process.wait()

    fusioned_filepath = narrow_swath_filepath.replace(extensions["narrow-swath"], extensions["fusioned_LSTs"])

    dc = read_datacube(fusioned_filepath)

    dc.compress(inplace=True)

    dc.save()

    return dc



def get_available_days(dtc_fitted_dc):

    nan_fraction = dtc_fitted_dc.data.LST_rebuilt.isnull().sum("time") / len(dtc_fitted_dc.data.time)
    always_nan = (nan_fraction > 0.9).values

    mask = (always_nan)[None, :, :] | np.isfinite(dtc_fitted_dc.data.LST_rebuilt.values)
    mask = xr.DataArray(mask.all(axis=(1, 2)), dims=("time"), coords={"time": dtc_fitted_dc.data.time})

    filtered_dc = copy.deepcopy(dtc_fitted_dc)
    filtered_dc.data = dtc_fitted_dc.data.where(mask, drop=True)

    available_days = np.unique([pd.to_datetime(d).date()
                                for d in filtered_dc.data.diurnal_cycle.values])

    return available_days