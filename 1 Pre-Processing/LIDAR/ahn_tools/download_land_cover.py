# noinspection PyUnusedImports
import rioxarray as rxr
import xarray as xr
import numpy as np
from rasterio.io import MemoryFile
from owslib.wms import WebMapService

def get_land_cover(dataset,
                   url="https://service.pdok.nl/wur/landelijk-grondgebruik-nederland/wms/v1_0?request=GetCapabilities&service=WMS",
                   layer="lgn-actueel"):

    wms = WebMapService(url, version="1.3.0")

    img = wms.getmap(layers=[layer],
                     size=(len(dataset.x), len(dataset.y)),
                     bbox=dataset.rio.bounds(),
                     srs=f"EPSG:28992",
                     format='image/png')

    with MemoryFile(img) as memfile:
        rgb = rxr.open_rasterio(memfile)
        r = rgb.sel(band=1).astype("uint32")
        g = rgb.sel(band=2).astype("uint32")
        b = rgb.sel(band=3).astype("uint32")
        hex_band = (r << 16) + (g << 8) + b
        hex_str = hex_band.values
        hex_codes = np.char.add("#", np.char.mod("%06X", hex_str))

        water_hex = ["#2473FF", "#000099"]

        mineral_hex = ["#E6FB00", "#F5E10F", "#FFFF00"]

        artificial_hex = ["#46FFCF", "#FF0000", "#730000", "#761818", "#FF645A", "#000000", "#871B00", "#A80000",
                          "#B02300"]

        vegetation_hex = ["#006400", "#009900", "#33C800", "#3CEF45", "#5ABA40", "#73DF1F", "#750075", "#89D42B",
                          "#93AA00",
                          "#93D600", "#93FF00", "#969639", "#A42353", "#A8EF44", "#AC81A8", "#AD8B06", "#B03060",
                          "#B26600",
                          "#B6B639", "#E3FF70", "#E51F7F", "#E89919", "#FF00C5", "#FFA500", "#FFB3A8", "#FFC0CB"]

        vegetalised = np.full_like(hex_band, 0)
        mineralised = vegetalised.copy()
        water_mask = vegetalised.copy()
        artificial = vegetalised.copy()

        for i in range(hex_codes.shape[0]):
            for j in range(hex_codes.shape[1]):
                if hex_codes[i, j] in vegetation_hex:
                    vegetalised[i, j] = 1
                if hex_codes[i, j] in mineral_hex:
                    mineralised[i, j] = 1
                if hex_codes[i, j] in water_hex:
                    water_mask[i, j] = 1
                if hex_codes[i, j] in artificial_hex:
                    artificial[i, j] = 1

    dataset["vegetalized"] = xr.DataArray(vegetalised.astype("bool"),
                                          dims=('y', 'x'),
                                          coords={"y": dataset.y, "x": dataset.x},
                                          attrs={"standard_name": "Vegetation/grass pixels",
                                                 "units": "no units",
                                                 "plot_kwargs": {"cmap": "Greens",
                                                                 "add_colorbar": False
                                                                 }
                                                 }
                                          )
    dataset["mineralized"] = xr.DataArray(mineralised.astype("bool"),
                                          dims=('y', 'x'),
                                          coords={"y": dataset.y, "x": dataset.x},
                                          attrs={"standard_name": "Mineralized pixels",
                                                 "units": "no units",
                                                 "plot_kwargs": {"cmap": "Greys",
                                                                 "add_colorbar": False
                                                                 }
                                                 }
                                          )

    dataset["artificial"] = xr.DataArray(artificial.astype("bool"),
                                         dims=('y', 'x'),
                                         coords={"y": dataset.y, "x": dataset.x},
                                         attrs={"standard_name": "Artificial pixels",
                                                "units": "no units",
                                                "plot_kwargs": {"cmap": "Reds",
                                                                "add_colorbar": False
                                                                }
                                                }
                                         )

    return dataset