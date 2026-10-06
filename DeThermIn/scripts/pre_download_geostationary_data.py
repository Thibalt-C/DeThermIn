"""
This scripts aims to download the MTG-FCI LST data for the summer 2026 via LSA-SAF (EUMETSAT) facilities.
It replaces the GEO LOADER script that used MSG-SEVERI LST data.
It also allows to download the data for the whole area of interest (France, the Netherlands).
"""

import os
import pickle
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import xarray as xr
from tqdm import tqdm
from datetime import datetime
from typing import Optional
from geopandas import GeoDataFrame
from ..metadata import parameters, available_areas, start, end


def predownload(
        start : Optional[datetime] = start,
        end : Optional[datetime] = end,
        areas : Optional[GeoDataFrame] = available_areas,
        path : Optional[str] = parameters["geostationary_data_path"],
        freq : Optional[str] = "1h"
):
    """
    Pre-downloads the geostationary data for the given area of interest.

    Parameters:
    ----------
    start : `datetime.datetime`, optional
        lower time bound of the time interval considered
    end : `datetime.datetime`, optional
        higher time bound of the time interval considered
    areas : `geopandas.GeoDataFrame`, optional
        area(s) of interest. Must contain a column named "code".
        Each set of rows with a different code will be made available separately.
    path : string, optional
        filepath to the downloaded data. Must exist.
    freq : string, optional
        frequency of acquisition of the data. Default is "1h".
        Must be a multiple of "10min" for MTG data (2025 and later) and "15min" for MSG data (2024 and earlier).
    """

    is_MTG = start.year>=2025

    base_url = "https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/"
    collection_path = "MTG/MTLST/NATIVE/" if is_MTG else "MSG/MLSTv2/NETCDF/"

    areas_of_interest = {}

    for code in np.unique(areas["code"]):
        area = areas.loc[areas["code"] == code].reset_index(drop=True)

        areas_of_interest[code] = {
            "area": area,
            "bounds": area.geometry.union_all().bounds
        }

    for code in np.unique(areas["code"]):

        area = areas.loc[areas["code"]==code].reset_index(drop=True)

        areas_of_interest[code] = {
            "area": area,
            "bounds": area.geometry.union_all().bounds
        }

    datasets = {key: [] for key in areas_of_interest.keys()}

    USER = input("Enter your LSA-SAF (EUMETSAT) username (https://datalsasaf.lsasvcs.ipma.pt/): ")
    PASSWORD = input("Enter your password: ")

    date_range = pd.date_range(start, end, freq=freq)

    iterator = tqdm(date_range, desc="Downloading geostationary data")

    for date in iterator:
        date_str = date.strftime("%Y/%m/%d/")
        if is_MTG: # Meteosat Third Generation
            file = f"LSA-007_MTG_MTLST_MTG-FD_{date.strftime('%Y%m%d%H%M')}.nc"
        else: # Meteosat Second Generation
            file = f"NETCDF4_LSASAF_MSG_MLSTv2_MSG-Disk_{date.strftime('%Y%m%d%H%M')}.nc"

        file_url = f"{base_url}{collection_path}{date_str}{file}"
        os.makedirs(".temporary/", exist_ok=True)
        filepath = ".temporary/" + file

        try:
            with requests.get(file_url, auth=HTTPBasicAuth(USER, PASSWORD), stream=True) as r:
                    r.raise_for_status()
                    with open(filepath, "wb") as f:
                        for chunk in r.iter_content(8192):
                            f.write(chunk)

            dataset = xr.open_dataset(filepath).load()

            if ("lat" in dataset.coords) & ("lon" in dataset.coords):
                dataset = dataset.rename({"lat": "y", "lon": "x"})
                dataset = dataset.rio.write_crs("EPSG:4326")
                already_geographic = True
            else:
                geostationary_proj = dataset.geostationary.attrs
                crs = (
                    "+proj=geos "
                    f"+h={geostationary_proj['perspective_point_height']} "
                    f"+lon_0={geostationary_proj['longitude_of_projection_origin']} "
                    f"+lat_0={geostationary_proj['latitude_of_projection_origin']} "
                    f"+a={geostationary_proj['semi_major_axis']} "
                    f"+b={geostationary_proj['semi_minor_axis']} "
                    f"+sweep={geostationary_proj['sweep_angle_axis']} "
                )
                dataset.coords["x"] = dataset.x * geostationary_proj["perspective_point_height"]
                dataset.coords["y"] = dataset.y * geostationary_proj["perspective_point_height"]
                dataset = dataset.rio.write_crs(crs)
                already_geographic = False

            for key in areas_of_interest.keys():
                sample = dataset.rio.clip_box(*areas_of_interest[key]["bounds"], crs="EPSG:4326")
                datasets[key].append(sample)

            os.remove(filepath)

        except:
            print(f"Warning: Unable to download {file_url}")
            pass

    os.rmdir(".temporary/")

    start = date_range[0].strftime("%Y%m%d")
    end = date_range[-1].strftime("%Y%m%d")

    for key in areas_of_interest.keys():
        datasets[key] = xr.concat(datasets[key], dim="time")
        if not already_geographic:
            datasets[key] = datasets[key].rio.reproject("EPSG:4326")
        if is_MTG:
            datasets[key]["LST"] = datasets[key].LST - 273.15 # to degrees Celsius
        datasets[key].attrs["plot_kwargs"] = {"cmap": "RdYlBu_r"}
        datasets[key].attrs["standard_name"] = "MTG-FCI Land Surface Temperature" if is_MTG else "MSG-SEVERI Land Surface Temperature"
        datasets[key].attrs["units"] = "°C"
        os.makedirs(path, exist_ok=True)
        filepath = path + f"{key}-{start}-{end}.pkl"
        pickle.dump(datasets[key], open(filepath, "wb"))