from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import folium
import numpy as np
import json

with open("DeThermIn/workflows/info.json", "rb") as f:
    info = json.load(f)

geofeatures_path = info["geofeatures_path"]
data_blending_pickles = info["data_blending_pickles"]

def to_base64(data_array, cmap='grey', vmin=None, vmax=None):

    array = data_array.values

    if "plot_kwargs" in data_array.attrs.keys():
        if "cmap" in data_array.attrs["plot_kwargs"].keys():
            cmap = data_array.attrs["plot_kwargs"]["cmap"]
        if "vmin" in data_array.attrs["plot_kwargs"].keys():
            vmin = data_array.attrs["plot_kwargs"]["vmin"]
        if "vmax" in data_array.attrs["plot_kwargs"].keys():
            vmax = data_array.attrs["plot_kwargs"]["vmax"]

    fig, ax = plt.subplots(figsize=(1, 1), dpi=1)
    ax.axis('off')

    im = ax.imshow(array, cmap=cmap, aspect='auto',vmin=vmin, vmax=vmax)
    fig.canvas.draw()

    img_data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img_data.shape = (fig.canvas.get_width_height()[::-1] + (4,))

    plt.close(fig)

    valid_mask = ~np.isnan(array)
    min_val = np.nanmin(array)
    max_val = np.nanmax(array)

    if min_val == max_val:
        normalized = np.ones_like(array, dtype=float)
    else:
        normalized = (array - min_val) / (max_val - min_val)

    try:
        cmap = plt.get_cmap(cmap)
    except: # if from cmcrameri already formatted
        pass

    rgba_float = cmap(normalized)
    rgba_uint8 = (rgba_float * 255).astype(np.uint8)
    rgba_uint8[~valid_mask, 3] = 0

    img = Image.fromarray(rgba_uint8, "RGBA")

    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)

    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    path = f"data:image/png;base64,{img_str}"

    plt.close('all')

    return path


def add_dataset_layer(dataset, layer, map, cmap='grey', show=True):

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
                                              bounds = ((float(lat_min), float(lon_min)), (float(lat_max), float(lon_max))),
                                              interactive=True,
                                              name=name,
                                              show=show)

    image.add_to(map)

    return


def map_dataset(dataset):


    reproj = dataset.LST.rio.reproject(dst_crs="EPSG:4326")

    if reproj.y.max() >= 51.0: # weak condition, can be more robust using the geofeatures
        country = "NLD"
    else:
        country = "FRA"

    map = folium.Map(location=[reproj.y.mean(), reproj.x.mean()], zoom_start=13, tiles="cartodb positron")

    if country=='FRA':
        wms = folium.raster_layers.WmsTileLayer(url = "https://data.geopf.fr/wms-r",
                                                layers =["HR.ORTHOIMAGERY.ORTHOPHOTOS"],
                                                fmt = "image/jpeg",
                                                version="1.3.0",
                                                srs = "EPSG:4326",
                                                name = "Ortho-image",
                                                attr="Orthoimage ©IGN 2026")
        wms.add_to(map)

    if country=='NLD':
        wms = folium.raster_layers.WmsTileLayer(url="https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0",
                                                layers=["Actueel_orthoHR"],
                                                fmt="image/jpeg",
                                                srs="EPSG:4326",
                                                name="Ortho-image",
                                                attr="Orthoimage © Netherlands Space Agency"
                                                )
        wms.add_to(map)

    excluded_layers = ["mask", "blue", "green", "red", "NIR", "SWIR1", "SWIR2", "qc", "elevation", "azimuth"]
    default_showed = ["LST", "SR_LST", "illumination", "irradiance", "energy", "NDVI"]

    land_cover_ordered = np.array(["water_mask", "artificial", "vegetalized", "mineralized", "buildings_mask"])

    all_layers = np.array(list(dataset.keys()))
    filtered_layers = [layer for layer in all_layers if
                       layer not in excluded_layers and layer not in land_cover_ordered]
    layers = np.concatenate([filtered_layers, land_cover_ordered])

    for layer in layers:
        if layer not in excluded_layers:
            try:
                add_dataset_layer(dataset, layer, map, show=True if layer in default_showed else False)
            except:
                pass

    folium.LayerControl().add_to(map)

    return map