import xarray as xr
import cmcrameri.cm as cm


def get_ndvi(dataset):
    dataset["NDVI"] = xr.DataArray((dataset.NIR - dataset.red) / (dataset.NIR + dataset.red),
                                   attrs={"standard_name": "NDVI", "units": "no units",
                                          "description": "Normalized differential vegetation index",
                                          "plot_kwargs": {"cmap": "Greens"}
                                          }
                                   )

    return dataset


def get_ndwi(dataset):
    dataset["MNDWI"] = xr.DataArray((dataset.green - dataset.SWIR1) / (dataset.green + dataset.SWIR1),
                                   attrs={"standard_name": "MNDWI", "units": "no units",
                                          "description": "Modified Normalized differential water index",
                                          "plot_kwargs": {"cmap": "Blues"}
                                          }
                                   )

    return dataset


def get_endisi(dataset):
    mndwi = (dataset.green - dataset.SWIR1) / (dataset.green + dataset.SWIR1)
    beta = 2 * dataset.blue.mean() / ((dataset.SWIR1 / dataset.SWIR2).mean() + (mndwi ** 2).mean())
    b = beta * (dataset.SWIR1 / dataset.SWIR2 + mndwi ** 2)

    dataset["ENDISI"] = xr.DataArray((dataset.blue - b) / (dataset.blue + b),
                                     attrs={"standard_name": "ENDISI", "units": "no units",
                                            "description": "Enhanced Normalized Difference Impervious Surfaces Index",
                                            "plot_kwargs": {"cmap": cm.bam_r}
                                            }
                                     )

    return dataset