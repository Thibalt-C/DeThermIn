from PIL import Image
import io
import base64
import folium
from xarray import DataArray, Dataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional
from ..metadata import available_areas
from shapely.geometry import Point
from pyproj.transformer import Transformer

def to_base64(
        data_array : DataArray,
        cmap : Optional[Union[str, plt.Colormap]] = 'grey',
        vmin : Optional[float] = None,
        vmax : Optional[float] = None
) -> str :

    array = data_array.values

    if "plot_kwargs" in data_array.attrs.keys():
        if "cmap" in data_array.attrs["plot_kwargs"].keys():
            cmap = data_array.attrs["plot_kwargs"]["cmap"]
        if "vmin" in data_array.attrs["plot_kwargs"].keys():
            vmin = data_array.attrs["plot_kwargs"]["vmin"]
        if "vmax" in data_array.attrs["plot_kwargs"].keys():
            vmax = data_array.attrs["plot_kwargs"]["vmax"]

    valid_mask = np.isfinite(array)
    min_val = np.nanmin(array) if vmin is None else vmin
    max_val = np.nanmax(array) if vmax is None else vmax

    if min_val == max_val:
        normalized = np.ones_like(array, dtype=float)
    else:
        normalized = (array - min_val) / (max_val - min_val)

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)

    rgba_float = cmap(normalized)
    rgba_uint8 = (rgba_float * 255).astype(np.uint8)
    rgba_uint8[~valid_mask, 3] = 0

    img = Image.fromarray(rgba_uint8, "RGBA")

    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)

    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    path = f"data:image/png;base64,{img_str}"

    return path


def add_dataset_layer(
        dataset : Dataset,
        layer : str,
        folium_map : folium.Map,
        cmap : Optional[Union[str, plt.Colormap]] = 'grey',
        show : Optional[bool] = True
):

    data_array = dataset[layer]

    if data_array.dtype == 'bool':
        data_array = data_array.astype("float32")
        data_array = data_array.where(data_array, np.nan)

    if "mask" in dataset.data_vars:
        data_array = data_array.where(dataset.mask)

    reprojected_data_array = data_array.rio.reproject(dst_crs="EPSG:4326")

    lat_min = reprojected_data_array.y.max().values.astype(np.float32)
    lat_max = reprojected_data_array.y.min().values.astype(np.float32)
    lon_min = reprojected_data_array.x.min().values.astype(np.float64)
    lon_max = reprojected_data_array.x.max().values.astype(np.float32)

    img = to_base64(reprojected_data_array, cmap)
    try:
        name = dataset[layer].attrs["standard_name"]
    except:
        name = layer

    min_val = np.nanmin(reprojected_data_array.values)
    max_val = np.nanmax(reprojected_data_array.values)

    try:
        units = " " + dataset[layer].attrs["units"]
    except:
        units = ""
    units = units.replace(" n.u.", "").replace(" no units", "")

    if min_val == max_val:
        name = name
    else:
        name = f"{name} [{min_val:.2f}, {max_val:.2f}{units}]"

    image = folium.raster_layers.ImageOverlay(image=img,
                                              bounds = ((float(lat_min), float(lon_min)),
                                                        (float(lat_max), float(lon_max))),
                                              interactive=True,
                                              name=name,
                                              show=show)

    image.add_to(folium_map)

    return


def map_dataset(dataset : Dataset) -> folium.Map :

    x, y = float(dataset.x.mean()), float(dataset.y.mean())
    lon, lat = Transformer.from_crs(dataset.rio.crs, "EPSG:4326", always_xy=True).transform(x,y)
    zones = available_areas.where(available_areas.geometry.intersects(Point(lon,lat))).dropna().reset_index(drop=True)
    country = np.unique(zones["country"].to_numpy())[0]

    folium_map = folium.Map(location=[lat, lon], zoom_start=13)

    if country=='FRA':
        wms = folium.raster_layers.WmsTileLayer(url = "https://data.geopf.fr/wms-r",
                                                layers ="HR.ORTHOIMAGERY.ORTHOPHOTOS",
                                                fmt = "image/jpeg",
                                                version="1.3.0",
                                                srs = "EPSG:4326",
                                                name = "Ortho-image",
                                                attr="Orthoimage ©IGN 2026")
        wms.add_to(folium_map)

    if country=='NLD':
        wms = folium.raster_layers.WmsTileLayer(url="https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0",
                                                layers="Actueel_orthoHR",
                                                fmt="image/jpeg",
                                                srs="EPSG:4326",
                                                name="Ortho-image",
                                                attr="Orthoimage © Netherlands Space Agency"
                                                )
        wms.add_to(folium_map)

    excluded_layers = ["mask", "blue", "green", "red", "NIR", "SWIR1", "SWIR2",
                       "qc", "elevation", "azimuth",
                       "LST_sunrise", "LST_next_sunrise",
                       "thermal_zenith", "thermal_sunset",
                       "LST_max", "diurnal_cycle"]
    default_showed = ["coarse_LST", "fine_LST", "SR_LST", "illumination", "irradiance", "energy", "NDVI"]

    land_cover_ordered = np.array(["water_mask", "artificial", "vegetalized", "mineralized", "buildings_mask"])

    all_layers = np.array(list(dataset.keys()))
    filtered_layers = [layer for layer in all_layers if
                       layer not in excluded_layers and layer not in land_cover_ordered]
    layers = np.concatenate([filtered_layers, land_cover_ordered])

    for layer in layers:
        if layer not in excluded_layers:
            try:
                add_dataset_layer(dataset, layer, folium_map, show=True if layer in default_showed else False)
            except:
                pass

    folium.LayerControl().add_to(folium_map)

    return folium_map