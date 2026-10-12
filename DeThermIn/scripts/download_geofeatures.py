"""
This script aims to generate the geofile containing all the areas with available data.
It uses geodata from the French and Dutch administrative boundaries.
"""

import pandas as pd
from owslib.wms import WebMapService
import geopandas as gpd
import os
from owslib.wfs import WebFeatureService
import json
import numpy as np
from ..metadata import metadata

path = metadata["geofeatures_path"]
filepath = os.path.join(path, metadata["available_areas_file"])

def download_available_areas():

    # Getting Dutch cities

    wms = WebMapService("https://opengeodata.zeeland.nl/geoserver/zeeland/wms?request=GetCapabilities")

    gpkg = wms.getmap(
        layers=["geogrz_gemgrznldvlk"],
        format="kmz",
        crs="EPSG:28992",
        bbox=wms["geogrz_gemgrznldvlk"].boundingBox[:4],
        size=(1024,1024)
    )

    # creates the folder if it doesn't exist
    os.makedirs(path, exist_ok=True)

    with open(path+'Dutch_cities.kmz', 'wb') as f:
        f.write(gpkg.read())

    nl = gpd.read_file(path+'Dutch_cities.kmz')
    nl["City"] = None
    for i in range(len(nl)):
        element = nl.loc[i, "description"]
        city = element.split('class="atr-name">gm_naam</span>:</strong> <span class="atr-value">')[-1]
        city = city.split("</span>")[0]
        nl.loc[i, "City"]= city

    os.remove(path+'Dutch_cities.kmz')

    print("Dutch cities imported.")


    # Getting French cities

    print("(About 35000 French cities to get)")

    wfs = WebFeatureService(
        "https://data.geopf.fr/wfs/ows?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetCapabilities",
        version="2.0.0"
    )

    layer = "LIMITES_ADMINISTRATIVES_EXPRESS.LATEST:commune"

    count = 5000
    start = 0
    features = []

    while True:
        response = wfs.getfeature(
            typename=[layer],
            outputFormat="application/json",
            startindex=start
        )

        data = json.load(response)

        current_features = data["features"]

        if not current_features:
            break

        features.extend(current_features)

        print(f"Retrieved: {len(features)}")

        if len(current_features) < count:
            break

        start += count

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(path+"French_cities.geojson", "w") as f:
        json.dump(geojson, f)

    fr = gpd.read_file(path+"French_cities.geojson")

    fr["com_area_co"] = None
    fr.loc[fr["code_insee_de_la_region"] == "04", "com_area_co"] = "REU"
    fr.loc[fr["code_insee_de_la_region"] == "01", "com_area_co"] = "GLP"
    fr.loc[fr["code_insee_de_la_region"] == "02", "com_area_co"] = "MTQ"
    fr.loc[fr["code_insee_de_la_region"] == "06", "com_area_co"] = "MYT"
    fxx = np.array([
        (fr["code_insee_de_la_region"] == code) for code in ['11', '24', '27', '28', '32', '44', '52', '53', '75', '76', '84', '93', '94']
    ]).any(axis=0)
    fr.loc[fxx, "com_area_co"] = "FXX"

    os.remove(path+"French_cities.geojson")

    print("French cities imported.")


    # Creating the geofile containing all the areas with available data

    regions = []

    for code in ["REU", "MYT", "GLP", "MTQ", "FXX"]:
        region = fr.loc[fr["com_area_co"]==code]
        cities = region["nom_officiel"]
        geom = region["geometry"]
        region = gpd.GeoDataFrame({"code": code, "name": cities, "geometry": geom, "country": "FRA"})
        regions.append(region)

    nl = gpd.GeoDataFrame({"code": "NL", "name": nl["City"], "geometry": nl["geometry"], "country": "NLD"})
    regions.append(nl)

    available = gpd.GeoDataFrame(pd.concat(regions), crs="EPSG:4326")
    available.to_file(filepath, driver="GeoJSON", index=False)