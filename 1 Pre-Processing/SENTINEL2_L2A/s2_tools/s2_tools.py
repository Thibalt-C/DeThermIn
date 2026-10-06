import rioxarray as rxr
import xarray as xr
import os
import datetime
import numpy as np

def prepare_request(connection, gdf, start, end, size, collection="SENTINEL2_L2A"):

    #gdf_bounds = gdf.union_all().bounds
    gdf_center = gdf.union_all().centroid.coords[0]
    lon, lat = gdf_center
    d_lon = size / (111320 * np.cos(np.radians(lat)))
    d_lat = size / 111320
    gdf_bounds = [lon - d_lon/2, lat - d_lat/2, lon + d_lon/2, lat + d_lat/2]

    start = datetime.datetime.strptime(start, "%m-%d-%Y")
    start = start.strftime("%Y-%m-%d")
    end = datetime.datetime.strptime(end, "%m-%d-%Y")
    end = end.strftime("%Y-%m-%d")

    datacube = connection.load_collection(collection_id=collection,
                                          spatial_extent={"west": gdf_bounds[0], "south": gdf_bounds[1],
                                                          "east": gdf_bounds[2], "north": gdf_bounds[3]},
                                          temporal_extent=[start, end],
                                          bands=["B02", "B03", "B04", # B, G, R
                                                 "B08", # NIR
                                                 "B11", "B12", # SWIR
                                                 "SCL"] # Scene classification
                                          )
    datacube = datacube.mask((datacube.band("SCL") == 8) |
                             (datacube.band("SCL") == 9) |
                             (datacube.band("SCL") == 10) |
                             (datacube.band("SCL") == 3))  # cloud mask
    datacube = datacube.mean_time()  # time averaging
    datacube = datacube.filter_bands(["B02", "B03", "B04", "B08", "B11", "B12"])  # keep these bands
    result = datacube.save_result("GTiff")  # save as a geotiff

    return result


def request_and_download(result, name, folder="data"):

    job = result.create_job(title=name)
    job.start_and_wait()
    filepath = os.path.join(folder, name+".tif")
    job.get_results().download_file(filepath)

    return filepath


def create_s2_dataset(filepath):

    S2 = rxr.open_rasterio(filepath)
    S2 = S2.load()
    crs = S2.rio.crs

    blue = S2.sel(band=1).drop_vars("band")
    blue.attrs = {"standard_name": "Blue", "units": "W/m^2/sr/µm", "long_name":"Blue band radiance (442 ±20 nm)", "plot_kwargs": {"cmap": "Blues"}}
    green = S2.sel(band=2).drop_vars("band")
    green.attrs = {"standard_name": "Green", "units": "W/m^2/sr/µm", "long_name": "Green band radiance (492 ±64 nm)", "plot_kwargs": {"cmap": "Greens"}}
    red = S2.sel(band=3).drop_vars("band")
    red.attrs = {"standard_name": "Red", "units": "W/m^2/sr/µm", "long_name":"Red band radiance (558 ±35 nm)", "plot_kwargs": {"cmap": "Reds"}}
    nir = S2.sel(band=4).drop_vars("band")
    nir.attrs = {"standard_name": "NIR", "units": "W/m^2/sr/µm", "long_name":"Near infrared band radiance (832 ±118 nm)", "plot_kwargs": {"cmap": "Reds"}}
    swir1 = S2.sel(band=5).drop_vars("band")
    swir1.attrs = {"standard_name": "SWIR1", "units": "W/m^2/sr/µm", "long_name":"Short-wave infrared band radiance (1610 ±90 nm)", "plot_kwargs": {"cmap": "Reds"}}
    swir2 = S2.sel(band=6).drop_vars("band")
    swir2.attrs = {"standard_name": "SWIR2", "units": "W/m^2/sr/µm", "long_name": "Short-wave infrared band radiance (2190 ±180 nm)", "plot_kwargs": {"cmap": "Reds"}}

    S2_dataset = xr.Dataset(
        data_vars={"blue": blue, "green": green, "red": red, "NIR": nir, "SWIR1": swir1, "SWIR2": swir2},
        coords = {"x": S2.x, "y": S2.y}
    )

    S2_dataset.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    S2_dataset.rio.write_crs(crs, inplace=True)
    S2_dataset.rio.write_coordinate_system(inplace=True)


    return S2_dataset


