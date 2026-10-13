import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import cartopy.mpl.geoaxes
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.patheffects as path_effects
from IPython import get_ipython

matplotlib.rcParams['animation.embed_limit'] = 0.5 * 1e9 # 500 MB


def get_projection(crs):
    epsg = crs.to_epsg()
    if epsg == 4326:
        projection = ccrs.PlateCarree()
    else:
        projection = ccrs.epsg(epsg)
    return projection

def plot_dataset(dataset, layer, ax=None, gridlines=True, **kwargs):

    if layer not in dataset.data_vars:
        raise Exception(f"Layer {layer} not found in dataset.")

    if dataset[layer].dims != ('y','x'):
        raise Exception(f"Layer {layer} is not a spatial layer. Reduce the other dimensions first.")

    if dataset.rio.crs is None:
        raise Exception("Dataset has no CRS. Use `dataset.rio.write_crs()` to set the CRS.")

    projection = get_projection(dataset.rio.crs)

    if ax==None:
        fig, ax = plt.subplots(subplot_kw={"projection": projection}, figsize=(20, 8))
    elif (type(ax) != cartopy.mpl.geoaxes.GeoAxesSubplot) | (type(ax) != cartopy.mpl.geoaxes.GeoAxes):
        raise Exception("ax must be a `cartopy.mpl.geoaxes.GeoAxesSubplot` or `cartopy.mpl.geoaxes.GeoAxes`.")

    if "plot_kwargs" in dataset[layer].attrs:
        plot_kwargs = dataset[layer].plot_kwargs.copy()
    else:
        plot_kwargs = {}

    plot_kwargs.update(kwargs)

    to_plot = dataset[layer].copy()

    if to_plot.dtype == 'bool':
        to_plot = to_plot.where(to_plot, np.nan) # mask False values

    if "mask" in dataset.data_vars:
        to_plot = to_plot.where(dataset.mask)

    quadmesh = to_plot.plot(ax=ax, **plot_kwargs)

    if gridlines:
        gl = ax.gridlines(
            draw_labels=True,  # Enable labels for the gridlines
            linewidth=1,  # Line width of the gridlines
            color='black',  # Color of the gridlines
            alpha=0.5,  # Transparency of the gridlines
            linestyle='--',  # Line style of the gridlines
        )
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

    ax.axis('off')

    plt.tight_layout()

    return ax, quadmesh


def plot_orthophoto(dataset, ax=None):

    projection = get_projection(dataset.rio.crs)

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": projection}, figsize=(20, 8))

    dataset.orthoimage.plot.imshow(ax=ax, robust=True)

    ax.axis('off')

    return ax


def plot_timeseries(
        dataset,
        attribute,
        gridlines=True,
        orthoimage_background=True,
        savefig=None,
        **kwargs
):

    dataset = dataset.where(dataset[attribute].notnull(), drop=True)

    projection = get_projection(dataset.rio.crs)

    fig, ax = plt.subplots(subplot_kw={"projection": projection}, figsize=(15, 12))

    def update(frame, ax=ax):
        ax.clear()
        if orthoimage_background:
            ax = plot_orthophoto(dataset.isel(time=0), ax=ax)
        image = dataset.isel(time=frame)
        ax, quadmesh = plot_dataset(image, attribute, ax=ax, gridlines=gridlines, add_colorbar=False,
                                    vmin=dataset[attribute].min(), vmax=dataset[attribute].max(),
                                    **kwargs)
        ax.set_title(f"{pd.to_datetime(image.time.values).strftime('%d-%m-%Y %H:%M')}", fontsize=25)
        return ax

    ax, quadmesh = plot_dataset(dataset.isel(time=0), attribute, ax=ax, gridlines=gridlines,
                                vmin=dataset[attribute].min(), vmax=dataset[attribute].max(),
                                **kwargs)
    if ("standard_name" in dataset[attribute].attrs) & ("units" in dataset[attribute].attrs):
        label = f"{dataset[attribute].attrs["standard_name"]} [{dataset[attribute].attrs['units']}]"
    else:
        label = attribute
    quadmesh.colorbar.set_label(label, fontsize=16, labelpad=-10)
    quadmesh.colorbar.set_ticks([dataset[attribute].min(), dataset[attribute].max()])
    plt.draw()
    ani = FuncAnimation(fig, update, frames=len(dataset.time), interval=500)
    plt.tight_layout()
    if savefig:
        ani.save(savefig, writer="pillow", fps=5)
    if get_ipython() is not None:
        display(HTML(ani.to_jshtml()))
        plt.close(fig)
    else:
        plt.show()

    return


def add_scale(ax, datacube, scale_size=200, div_factor=20):
    resol = datacube.spatial_resolution
    n_pixels = int(scale_size//resol)
    x = datacube.data.x
    y = datacube.data.y
    center_x = len(x)//div_factor
    lower_y = len(y)//div_factor
    y_idxs = [-lower_y , -lower_y]
    x_idxs = [center_x, center_x+n_pixels]
    line = ax.plot(x[x_idxs], y[y_idxs], "-", color="w", linewidth=10)
    text = ax.text(x[x_idxs[1]], y[y_idxs[1]], f"   {scale_size} m", ha="left", va="center", fontweight="bold", color="w", fontsize=12)
    text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground="black"),
            path_effects.Normal()
        ])
    line[0].set_path_effects([
            path_effects.Stroke(linewidth=13, foreground="black"),
            path_effects.Normal()
        ])
    return