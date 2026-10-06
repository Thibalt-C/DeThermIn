import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs


amersfoort = ccrs.Stereographic(
    central_longitude=5.38763888888889,
    central_latitude=52.1561605555556,
    false_easting=155000,
    false_northing=463000,
    scale_factor=0.9999079,
    globe=ccrs.Globe(semimajor_axis=6377397.155, semiminor_axis=6356078.963)  # Bessel 1841
)



def plot_dataset(dataset,attribute, ax=None, projection=ccrs.PlateCarree()):

    if ax==None:
        fig, ax = plt.subplots(subplot_kw={"projection": projection}, figsize=(20, 8))

    dataset[attribute].plot(ax=ax,
                                  transform=amersfoort,
                                  **dataset[attribute].plot_kwargs
                                  )

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

    return ax