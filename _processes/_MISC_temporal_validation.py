import requests
import time
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from DeThermIn import read_datacube
from DeThermIn.metadata import start, end, extensions
from DeThermIn.scripts.utils import select_area_of_interest, get_geostationary_path
from DeThermIn.workflows import pre_process, DTC_fitting, spatio_temporal_fusion, super_resolution
import sys

path = sys.argv[1]

size = 500
resolution = 0.5 # higher spatial resolution
bbox_size = 5000 # smaller to process quicker

print("Please first update your token in the script. Consult https://portail-api.meteofrance.fr/web/fr/api/DonneesPubliquesClimatologie")
token = ""

base_url = "https://public-api.meteofrance.fr/public/DPClim/v1"

# stations in Paris containing hourly measurements of ground temperature
weather_stations = ["75114001"]

headers = {"Accept": "*/*", "Authorization": f"Bearer {token}"}

coords = []
files = []

for station in weather_stations:
    url = f"{base_url}/information-station?id-station={station}"
    with requests.get(url, headers=headers) as response:
        metadata = response.json()[0]
        name = metadata["nom"]
    lat = metadata["positions"][0]["latitude"]
    lon = metadata["positions"][0]["longitude"]
    coords.append((lon, lat))
    print(f"Downloding data of station {name} : {lat} {lon}...")

    start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")
    url = f"{base_url}/commande-station/horaire?id-station={station}&date-deb-periode={start_str}&date-fin-periode={end_str}"
    with requests.get(url, headers=headers) as response:
        command_id = response.json()["elaboreProduitAvecDemandeResponse"]["return"]

    url = f"{base_url}/commande/fichier?id-cmde={command_id}"
    time.sleep(10)
    file = f"{name}.csv"
    files.append(file)
    with requests.get(url, headers=headers, stream=True) as response:
        with open(file, "wb") as f:
            f.write(response.content)

for coord, file in zip(coords, files):

    lon, lat = coord
    lat, lon, city, region, country, size, AOI = select_area_of_interest(lat=lat, lon=lon, size=size)

    geos_path = get_geostationary_path(region)

    params = [AOI, start, end, resolution, bbox_size, geos_path]
    ancillary_dc, narrow_swath_dc, geostationary_dc = pre_process(*params)

    DeThermIn_path = geostationary_dc.filepath.replace(extensions["geostationary"], extensions["end_product"])

    if os.path.exists(DeThermIn_path):
        dc = read_datacube(DeThermIn_path)

    else:
        dtc_fitted_dc = DTC_fitting(geostationary_dc)
        fusioned_dc = spatio_temporal_fusion(narrow_swath_dc, dtc_fitted_dc)
        dc = super_resolution(ancillary_dc, fusioned_dc, mode="dynamic")

    df = pd.read_csv(file, sep=";")
    df["TCHAUSSEE"] = df["TCHAUSSEE"].str.replace(",", ".").astype(float)
    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d%H")
    df["DATE"] = df["DATE"] - datetime.timedelta(hours=0.5)

    timestamps = [pd.to_datetime(dc.data.time.values[0]), pd.to_datetime(dc.data.time.values[-1])]

    sample = df.loc[
        (df["DATE"] >= timestamps[0])
        & (df["DATE"] <= timestamps[-1])
    ].reset_index(drop=True)


    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    fig.subplots_adjust(hspace=0.1)

    axes[0].scatter(sample["DATE"], sample["TCHAUSSEE"], marker="+",
                    c="k", label="station's measured \nground temperature")
    dc.data.SR_LST.rio.reproject("EPSG:4326").sel(x=lon, y=lat, method="nearest").plot(ax=axes[0], x="time", c="r",
                                                                                       label="LST @0.5m")
    dc.data.fine_LST.rio.reproject("EPSG:4326").sel(x=lon, y=lat, method="nearest").plot(ax=axes[0], x="time", c="g",
                                                                                         label="LST @70m")

    dc.data.SR_LST.rio.reproject("EPSG:4326").quantile(0.1, ("x", "y")).plot(label="1st/9th decile LST @0.5m", c="k",
                                                                             ls="--", ax=axes[1])
    dc.data.SR_LST.rio.reproject("EPSG:4326").median(("x", "y")).plot(label="median LST @0.5m", c="k", ax=axes[1])
    dc.data.SR_LST.rio.reproject("EPSG:4326").quantile(0.9, ("x", "y")).plot(c="k", ls="--", ax=axes[1])

    dc.data.fine_LST.rio.reproject("EPSG:4326").quantile(0.1, ("x", "y")).plot(label="1st/9th decile LST @70m", c="g",
                                                                               ls="--", ax=axes[1])
    dc.data.fine_LST.rio.reproject("EPSG:4326").median(("x", "y")).plot(label="median LST @70m", c="g", ax=axes[1])
    dc.data.fine_LST.rio.reproject("EPSG:4326").quantile(0.9, ("x", "y")).plot(c="g", ls="--", ax=axes[1])

    for ax in axes:
        ax.set_xticks(pd.to_datetime(dc.data.time.values))
        ax.tick_params(axis="x", rotation=90)
        ax.set_xlabel("")
        ax.set_ylabel("Land Surface Temperature [°C]")
        ax.set_title("")
        ax.set_xlim(timestamps)
        ax.set_ylim([20, 70])
        ax.grid(True)
        ax.legend(loc="upper right")

    fig.savefig(path+f"temporal_validation_{name.replace(" ", "_")}.png", bbox_inches="tight", dpi=300)
