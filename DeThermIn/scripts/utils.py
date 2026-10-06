

import os
from ..metadata import parameters, available_areas, extensions
import datetime
from shapely.geometry import Point, Polygon
import geopandas as gpd
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt
from owslib.wms import WebMapService
import requests
import time


def bold_text(text):
    return "\033[1m" + text + "\033[0m"


def get_processed_areas():

    if not os.path.exists(parameters["processed_data_path"]):
        raise FileNotFoundError()

    processed_areas = {}
    filenames = {}
    regions = {}
    zones = {}

    for file in os.listdir(parameters["processed_data_path"]):

        if file.endswith(".pkl"):

            lon = int(file.split("_")[0]) + 0.0001 * int(file.split("_")[1])
            lat = int(file.split("_")[2]) + 0.0001 * int(file.split("_")[3])

            start_month = int(file.split("_")[4])
            start_day = int(file.split("_")[5])
            start_year = int(file.split("_")[6])

            end_month = int(file.split("_")[7])
            end_day = int(file.split("_")[8])
            end_year  = int(file.split("_")[9])

            status = file.split("_")[10].split(".")[0]

            start_date = datetime.date(start_year, start_month, start_day)
            end_date = datetime.date(end_year, end_month, end_day)

            aoi = available_areas.where(available_areas.contains(Point(lon, lat))).dropna().reset_index(drop=True)

            key = f"{lat}°N {lon}°E, {aoi.loc[0, "name"]} ({aoi.loc[0, "country"]}) {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

            if key in processed_areas:
                processed_areas[key].append(status)
                filenames[key] = file.split(status)[0][:-1]
                regions[key] = aoi.loc[0, "code"]
                zones[key] = aoi.loc[0, "country"]
            else:
                processed_areas[key] = [status]

    return processed_areas, filenames, regions, zones


def select_processed_area(processed_areas, filenames, regions, zones):

    if len(processed_areas) != 0:

        print("Some data has been already (fully or partially) processed:")

        status_list = {}
        valid_keys = []

        n = 0
        for key in processed_areas.keys():

            processing_status = None
            if extensions["lidar"] in processed_areas[key]:
                processing_status = "partially pre-processed (0/4)"
            if (
                    (extensions["geostationary"] in processed_areas[key])
                    & (extensions["lidar"] in processed_areas[key])
                    & (extensions["narrow-swath"] in processed_areas[key])
                    & (extensions["blended_ancillary_data"] in processed_areas[key])
            ):
                processing_status = "pre-processed (1/4)"
            if (extensions["DTC-fitted"] in processed_areas[key]):
                processing_status = "DTC_model-fitted (2/4)"
            if (extensions["fusioned_LSTs"] in processed_areas[key]):
                processing_status = "fusioned (3/4)"
            if (extensions["end_product"] in processed_areas[key]):
                processing_status = "processed (4/4)"

            if processing_status is not None:
                n += 1; print(f" [{bold_text(str(n))}] {key}: {processing_status}")
                status_list[key] = processing_status
                valid_keys.append(key)

        selection = input("Type the number corresponding to the area of interest you want to select."
                          +f"\nType [{bold_text("S")}] to skip and enter coordinates manually.\n")

    else:
        selection = "s"

    if selection.lower() != "s":
        if selection.isdigit():
            if int(selection) > len(status_list):
                raise Exception(f"This number is out of the range.")
            else:
                processing_status = status_list[valid_keys[int(selection)-1]]
                filename = filenames[valid_keys[int(selection)-1]]
                region = regions[valid_keys[int(selection)-1]]
                zone = zones[valid_keys[int(selection)-1]]
                from_scratch = False
        else:
            raise Exception("This is not a valid entry.\n")
    else:
        from_scratch = True
        filename = None
        processing_status = None


    return processing_status, filename, from_scratch, region, zone



def select_area_of_interest():

    lat = float(input("\nProvide the latitude of the area of interest: "))
    lon = float(input("Provide the longitude of the area of interest: "))

    aoi = available_areas.where(available_areas.contains(Point(lon, lat))).dropna().reset_index(drop=True)

    if len(aoi) == 0:
        raise Exception("This location is out of the available areas. Please choose another one.")
    else:
        city = aoi.loc[0, "name"]
        region = aoi.loc[0, "code"]
        country = aoi.loc[0, "country"]
        print(f"You selected a point located in {bold_text(city)} ({bold_text(country)}).")

    size = 0
    while (size < 500) | (size > 10000):
        size = int(
            input("Provide the size of the area of interest around the given coordinates (between 500 and 10000 m): "))

    print("The area of interest covers the area shown in the following image:")

    lat_degree = size / 111320.0
    lon_degree = size / (111320.0 * abs(np.cos(np.radians(lat))))

    top = lat + lat_degree / 2
    bottom = lat - lat_degree / 2
    right = lon + lon_degree / 2
    left = lon - lon_degree / 2

    polygon = Polygon([
        (left, top),
        (right, top),
        (right, bottom),
        (left, bottom)
    ])

    AOI = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")

    if country == "NLD":
        wms = WebMapService("https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0?&request=GetCapabilities&service=wms",
                            version="1.3.0")
        layer = "Actueel_orthoHR"
    elif country == "FRA":
        wms = WebMapService("https://data.geopf.fr/wms-r?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities",
                            version="1.3.0")
        layer = "HR.ORTHOIMAGERY.ORTHOPHOTOS"

    img = wms.getmap(layers=[layer],
                     size=(500, 500),
                     bbox=[left, bottom, right, top],
                     srs=f"EPSG:4326",
                     format='image/jpeg'
                     )

    orthophoto = Image.open(io.BytesIO(img.read()))
    orthophoto = np.array(orthophoto)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")
    plt.imshow(orthophoto)
    plt.show()

    input("\nPress Enter to continue with this area of interest...")

    return lat, lon, city, region, country, size, AOI


def check_requirements():

    print("\nWaiting for internet connection...")
    while True:
        try:
            requests.head("https://www.wikipedia.org/", timeout=1)
            break
        except requests.ConnectionError:
            time.sleep(3)

    print("-> connected to internet.")

    if not os.path.exists(parameters["processed_data_path"]):
        print(f"Folder {parameters["processed_data_path"]} does not exist. Please create it to proceed.\n")
    while not os.path.exists(parameters["processed_data_path"]):
        time.sleep(3)

    if not os.path.exists(parameters["geostationary_data_path"]):
        print(f"Folder {parameters["geostationary_data_path"]} does not exist. Please create it to proceed.\n")
    while not os.path.exists(parameters["geostationary_data_path"]):
        time.sleep(3)

    print("-> all the required paths are existing.")