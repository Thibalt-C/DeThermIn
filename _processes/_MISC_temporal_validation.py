import requests
import time
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects

from DeThermIn import read_datacube
from DeThermIn.metadata import start, end, extensions
from DeThermIn.scripts.utils import select_area_of_interest, get_geostationary_path
from DeThermIn.workflows import pre_process, DTC_fitting, spatio_temporal_fusion, super_resolution
import sys

path = sys.argv[1]

resolution = 1
size = 1000
bbox_size = 5000 # smaller to process quicker

#token = input("Please enter your token from https://portail-api.meteofrance.fr/web/fr/api/DonneesPubliquesClimatologie: ")

base_url = "https://public-api.meteofrance.fr/public/DPClim/v1"

# stations in Paris containing hourly measurements of ground temperature
weather_stations = [
    "95527001,"
    #"75114001"
]

#headers = {"Accept": "*/*", "Authorization": f"Bearer {token}"}

coords = [(2.5343,49.0152)]
files = ["_processes/ROISSY-EN-FRANCE (ROISSY).csv"]
name = "ROISSY-EN-FRANCE (ROISSY)"

# for station in weather_stations:
#     url = f"{base_url}/information-station?id-station={station}"
#     with requests.get(url, headers=headers) as response:
#         metadata = response.json()[0]
#         name = metadata["nom"]
#     lat = metadata["positions"][0]["latitude"]
#     lon = metadata["positions"][0]["longitude"]
#     coords.append((lon, lat))
#     print(f"Downloding data of station {name} : {lat} {lon}...")
#
#     start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
#     end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")
#     url = f"{base_url}/commande-station/horaire?id-station={station}&date-deb-periode={start_str}&date-fin-periode={end_str}"
#     with requests.get(url, headers=headers) as response:
#         command_id = response.json()["elaboreProduitAvecDemandeResponse"]["return"]
#
#     url = f"{base_url}/commande/fichier?id-cmde={command_id}"
#     time.sleep(10)
#     file = f"_processes/{name}.csv"
#     files.append(file)
#     with requests.get(url, headers=headers, stream=True) as response:
#         with open(file, "wb") as f:
#             f.write(response.content)

for coord, file in zip(coords, files):

    lon, lat = coord
    lat, lon, city, region, country, size, AOI = select_area_of_interest(lat=lat, lon=lon, size=size)

    geos_path = get_geostationary_path(region)

    params = [AOI, start, end, resolution, bbox_size, geos_path]
    ancillary_dc, narrow_swath_dc, geostationary_dc = pre_process(*params)

    DeThermIn_path = geostationary_dc.filepath.replace(extensions["geostationary"], extensions["end_product"])
    fusioned_path = geostationary_dc.filepath.replace(extensions["geostationary"], extensions["fusioned_LSTs"])

    if os.path.exists(fusioned_path):
        fusioned_dc = read_datacube(fusioned_path)

    else:
        dtc_fitted_dc = DTC_fitting(geostationary_dc)
        fusioned_dc = spatio_temporal_fusion(narrow_swath_dc, dtc_fitted_dc)

    if os.path.exists(DeThermIn_path):
        dc = read_datacube(DeThermIn_path)
    else:
        dc = super_resolution(ancillary_dc, fusioned_dc, mode="dynamic")
        dc.save()
        dc2 = super_resolution(ancillary_dc, fusioned_dc, mode="static")
        dc.data["SR_LST_static"] = dc2.data["SR_LST_static"]
        dc.save()

    dc.data = dc.data.where(dc.data.SR_LST_dynamic.notnull(), drop=True)

    vmin = int(dc.data.SR_LST_dynamic.min().values)
    vmax = int(dc.data.SR_LST_dynamic.max().values)

    df = pd.read_csv(file, sep=";")
    df["TCHAUSSEE"] = df["TCHAUSSEE"].str.replace(",", ".").astype(float)
    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d%H")
    df["DATE"] = df["DATE"] - datetime.timedelta(hours=0.5)

    timestamps = [pd.to_datetime(dc.data.time.values[0]), pd.to_datetime(dc.data.time.values[-1])]

    sample = df.loc[
        (df["DATE"] >= timestamps[0])
        & (df["DATE"] <= timestamps[-1])
    ].reset_index(drop=True)


    fig = plt.figure(figsize=(17, 7))
    gs = gridspec.GridSpec(5,8, fig, height_ratios=[0.9, 0.1, 0.5, 0.5, 0.05])

    axes = [fig.add_subplot(gs[0, :6]),
            fig.add_subplot(gs[0, 6:])]

    space = fig.add_subplot(gs[1, :])
    space.axis("off")

    LST_static_axes = [fig.add_subplot(gs[2, 0]),
                       fig.add_subplot(gs[2, 1]),
                       fig.add_subplot(gs[2, 2]),
                       fig.add_subplot(gs[2, 3]),
                       fig.add_subplot(gs[2, 4]),
                       fig.add_subplot(gs[2, 5]),
                       fig.add_subplot(gs[2, 6]),
                       fig.add_subplot(gs[2, 7])]

    LST_dynamic_axes = [fig.add_subplot(gs[3, 0]),
                        fig.add_subplot(gs[3, 1]),
                        fig.add_subplot(gs[3, 2]),
                        fig.add_subplot(gs[3, 3]),
                        fig.add_subplot(gs[3, 4]),
                        fig.add_subplot(gs[3, 5]),
                        fig.add_subplot(gs[3, 6]),
                        fig.add_subplot(gs[3, 7])]

    cbar_ax = fig.add_subplot(gs[4, :])

    fig.subplots_adjust(hspace=0.1)

    axes[0].scatter(sample["DATE"], sample["TCHAUSSEE"], marker="+",
                    c="k", label="station's measured \nground temperature")
    dc.data.SR_LST_static.rio.reproject("EPSG:4326").sel(x=lon, y=lat, method="nearest").plot(ax=axes[0], x="time", c="r", ls="--",
                                                                                       label="LST @1m, set B")
    dc.data.SR_LST_dynamic.rio.reproject("EPSG:4326").sel(x=lon, y=lat, method="nearest").plot(ax=axes[0], x="time", c="r",
                                                                                       label="LST @1m, set C")
    dc.data.fine_LST.rio.reproject("EPSG:4326").sel(x=lon, y=lat, method="nearest").plot(ax=axes[0], x="time", c="g",
                                                                                         label="LST @70m")

    axes[0].set_xticks(pd.to_datetime(dc.data.time.values[::3]))
    axes[0].set_xticklabels(pd.to_datetime(dc.data.time.values[::3]).strftime("%H:%M"))
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Land Surface Temperature [°C]")
    axes[0].set_title(f"{pd.to_datetime(dc.data.time.values[0]).strftime('%Y-%m-%d')}, station {name} ({lat:.3f}°N, {lon:.3f}°E)")
    axes[0].set_xlim(timestamps)
    axes[0].set_ylim([15, 70])
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    dc.plot("orthoimage", ax=axes[1], scale_size=100)
    axes[1].set_title("")
    axes[1].scatter(dc.data.x[len(dc.data.x)//2], dc.data.y[len(dc.data.y)//2], marker="o", edgecolors="k", c="w", s=150)


    for n, ax in enumerate(LST_static_axes):
        dc.data.SR_LST_static.isel(time=n*3).plot(ax=ax, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, add_colorbar=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        if n==0:
            ax.set_ylabel("LST @1m, set B")
        ax.set_title(f"{pd.to_datetime(dc.data.time.values[n*3]).strftime('%H:%M UTC')}")

    for n, ax in enumerate(LST_dynamic_axes):
        im=dc.data.SR_LST_dynamic.isel(time=n*3).plot(ax=ax, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, add_colorbar=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        if n==0:
            ax.set_ylabel("LST @1m, set C")
        ax.set_title("")

    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Land Surface Temperature [°C]", fontsize=14, labelpad=-10)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([vmin, vmax])

    for ax, letter in zip(axes, ["a", "b"]):
        txt = ax.text(
            0.02, 0.95, letter,
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="top",
            color="black"
        )
        txt.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground="white"),
            path_effects.Normal()
        ])

        for ax, letter in zip([LST_static_axes[0], LST_dynamic_axes[0]], ["c", "d"]):
            txt = ax.text(
                0.02, 0.95, letter,
                transform=ax.transAxes,
                fontsize=18,
                fontweight="bold",
                va="top",
                color="black"
            )
            txt.set_path_effects([
                path_effects.Stroke(linewidth=2, foreground="white"),
                path_effects.Normal()
            ])

    fig.savefig(path+f"temporal_validation_{name.replace(" ", "_")}.png", bbox_inches="tight", dpi=300)