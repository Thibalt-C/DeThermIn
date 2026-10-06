"""
This script aims to generate the geofile containing all the areas with available data.
It uses geodata from the French and Dutch administrative boundaries.
"""

import pandas as pd
from owslib.wms import WebMapService
import geopandas as gpd
import os


# Getting Dutch cities

wms = WebMapService("https://opengeodata.zeeland.nl/geoserver/zeeland/wms?request=GetCapabilities")

gpkg = wms.getmap(
    layers=["geogrz_gemgrznldvlk"],
    format="kmz",
    crs="EPSG:28992",
    bbox=wms["geogrz_gemgrznldvlk"].boundingBox[:4],
    size=(1024,1024)
)

with open('Dutch_cities.kmz', 'wb') as f:
    f.write(gpkg.read())

nl = gpd.read_file('Dutch_cities.kmz')
nl["City"] = None
for i in range(len(nl)):
    element = nl.loc[i, "description"]
    city = element.split('class="atr-name">gm_naam</span>:</strong> <span class="atr-value">')[-1]
    city = city.split("</span>")[0]
    nl.loc[i, "City"]= city

os.remove('Dutch_cities.kmz')

print("Dutch cities imported.")


# Getting French cities

fr = gpd.read_file("communes-france")

print("French cities imported.")


# Creating the geofile containing all the areas with available data

regions = []

for code in ["REU", "MYT", "GLP", "MTQ", "FXX"]:
    region = fr.loc[fr["com_area_co"]==code]
    cities = region["com_name_up"]
    geom = region["geometry"]
    region = gpd.GeoDataFrame({"code": code, "name": cities, "geometry": geom, "country": "FRA"})
    regions.append(region)

nl = gpd.GeoDataFrame({"code": "NL", "name": nl["City"], "geometry": nl["geometry"], "country": "NLD"})
regions.append(nl)

available = gpd.GeoDataFrame(pd.concat(regions), crs="EPSG:4326")
available.to_file("available_areas.geojson", driver="GeoJSON", index=False)