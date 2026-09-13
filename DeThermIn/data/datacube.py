import cartopy.mpl.geoaxes
import pandas as pd
from DeThermIn.data.enricher import *
from DeThermIn.data.utils import *
from DeThermIn.data.visualisation import *
from DeThermIn.data.interactive import *
from copy import deepcopy
import os
import datetime
import pickle
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
import matplotlib
from typing import Optional, Union, Tuple


class DataCube:
    """
    Represents a data cube for handling and processing geospatial datasets.

    The DataCube class facilitates the management, processing, and interaction with
    geospatial datasets stored in the format of xarray Datasets. It provides functionality
    for dataset compression, visualization, fusion with other datasets, and enriching
    datasets with derived layers. Additionally, it supports interaction with attached
    datacubes and processing workflows.

    Attributes:
    ----------
    data : xr.Dataset
        The main geospatial dataset housed in the DataCube.

    filepath : str
        The path to the file corresponding to the dataset.

    units : dict
        Units of measurement for spatial and temporal data dimensions.

    path : str
        The directory path where the file is stored.

    spatial_resolution : float
        The spatial resolution of the dataset.

    temporal_resolution : float or None
        The temporal resolution of the dataset.

    status : str
        The current processing status of the datacube.

    name : str
        The name of the datacube derived from the file path and status.

    ngigabytes : float
        The size of the dataset in gigabytes.

    enriched : bool
        Indicates whether the dataset has been enriched with derived layers.
    """

    def __init__(
            self,
            data : xr.Dataset,
            filepath : str
    ):
        """
        Initialises the necessary attributes and processes related to the dataset and its filepath.

        Parameters:
        ----------
        data : xr.Dataset
            The dataset to initialise the object with.
        filepath : str
            Path to the corresponding file. Must be a .pkl file.
        """

        if not isinstance(filepath, str):
            raise Exception("Filepath must be a string.")
        elif not filepath.endswith(".pkl"):
            raise Exception("File must be a .pkl file.")
        if not isinstance(data, xr.Dataset):
            raise Exception("Data must be an xarray Dataset.")

        self.data = data
        self.filepath = filepath
        self.units = {"space": "m", "time": "h"}

        self.update()

        return


    def __repr__(self):
        temporal_res = f"Temporal resolution: {self.temporal_resolution:.0f} {self.units['time']}\n" if isinstance(self.temporal_resolution, float) else ""
        return (
            f"<DeThermIn.data.datacube.Datacube> {self.name} ({self.status})\n"
            "------------\n"
            f"Spatial resolution: {self.spatial_resolution:.2f} {self.units['space']}\n"
            f"{temporal_res}"
            "------------\n"
            f"Total size: {self.ngigabytes:.2f} GB\n"
            "------------\n"
            f"{str(self.data)}"
        )


    def update(self):
        """
        Updates the instance attributes based on the current state of the object.
        """

        self.path = os.path.dirname(self.filepath)
        self.spatial_resolution = find_spatial_resolution(self.data)
        self.temporal_resolution = find_temporal_resolution(self.data)
        self.status = find_status(self.filepath)
        self.name = find_name(self.filepath, self.status)
        self.ngigabytes = self.data.nbytes / 1e9
        self.enriched = True if (self.status == "ENRICHED") | (self.status == "TRAININGREADY") else False

        return


    def update_status(self, status):
        """
        Combines the current path, object name, and the given status to generate a
        new file path and updates it within the object.

        Parameters:
        ----------
        status : str
            New status to be assigned to the object.
        """

        self.filepath = os.path.join(self.path , f"{self.name}_{status}.pkl")
        self.update()

        return


    def compress(
            self,
            inplace: Optional[bool] = True
    ):
        """
        Compresses the dataset thanks to smaller float or int encodings and automatic conversion to boolean.

        The method applies a compression process to the dataset and modifies it based
        on the provided parameter. It is useful for reducing the size of the dataset
        by eliminating redundant or unnecessary data.

        Parameters:
        ----------
        inplace : bool, optional
            Whether to perform the operation in-place. Defaults to True.

        Returns:
        ----------
        DataCube
            Compressed datacube
        """

        dc = inplace_rule(self, inplace)

        dc.data = dataset_compressor(dc.data)

        return dc


    def plot(
            self,
            layer : str,
            show_scale : Optional[bool] = True,
            ax : Optional[Union[cartopy.mpl.geoaxes.GeoAxes, cartopy.mpl.geoaxes.GeoAxesSubplot]] = None,
            **kwargs
    ) -> Union[cartopy.mpl.geoaxes.GeoAxes, Tuple[cartopy.mpl.geoaxes.GeoAxes, matplotlib.collections.QuadMesh], None]:
        """
        Plots a specific layer from the datacube, supporting both spatial and temporal layers. This method allows visualization of
        orthophotos, spatial datasets, or time series data depending on the specified layer and its dimensions within the datacube.

        Parameters:
        ----------
        layer : str
            The name of the layer to be visualized. It must exist within the datacube as a data variable and have compatible dimensions.
        show_scale : bool, optional
            Whether to display a scale indicator on the plot (default is True).
        ax : cartopy.mpl.geoaxes.GeoAxes or cartopy.mpl.geoaxes.GeoAxesSubplot, optional
            A specific Cartopy GeoAxes to draw the plot onto. If None, a new axis will be created.
        kwargs : dict
            Optional keyword arguments for additional plot configuration or dimension reduction.

        Returns:
        ----------
        cartopy.mpl.geoaxes.GeoAxes or Tuple[cartopy.mpl.geoaxes.GeoAxes, matplotlib.collections.QuadMesh] or None
            The GeoAxes object with the plotted layer. If the layer is a spatial dataset, a QuadMesh is also returned. For
            time series plots, nothing is returned.
        """

        if layer not in self.data.data_vars:
            raise Exception(f"Layer {layer} not found in datacube.")

        if layer == "orthoimage":
            ax = plot_orthophoto(self.data, ax=ax)
            if show_scale:
                add_scale(ax, self)
            return ax

        elif (self.data[layer].dims != ('y','x')) & ('time' not in self.data[layer].dims):
            dims = self.data[layer].dims
            data = deepcopy(self.data)
            for dim in dims:
                if dim in kwargs.keys():
                    data = data.sel({dim: kwargs[dim]}, method="nearest")
                    kwargs.__delitem__(dim)
            if data[layer].dims != ('y','x'):
                raise Exception(f"Layer {layer} is not a spatial layer. Reduce the other dimensions.")
            else:
                ax, quadmesh = plot_dataset(data, layer, ax=ax, **kwargs)
                if show_scale:
                    add_scale(ax, self)
                return ax, quadmesh

        elif 'time' in self.data[layer].dims:
            plot_timeseries(self.data, layer, **kwargs)
            return

        else:
            ax, quadmesh = plot_dataset(self.data, layer, ax=ax, **kwargs)
            if show_scale:
                add_scale(ax, self)
            return ax, quadmesh


    def make_interactive_map(
            self,
            timestamp : Optional[datetime.datetime] = None,
            save : Optional[bool] = True
    ) -> folium.Map:
        """
        Generates an interactive map for a specific timestamp and optionally saves it as an HTML file.

        Parameters:
        ----------
        timestamp : datetime.datetime, optional
            The timestamp for which the data should be used to generate the map. If not provided,
            the earliest available timestamp in the dataset will be used.
        save : bool, optional
            If True, saves the generated map as an HTML file. The filename will include the selected
            timestamp. Default is True.

        Returns:
        ----------
        folium.Map
            The generated interactive map.
        """

        if timestamp is None:
            timestamp = self.data.time.values[0]

        map = map_dataset(self.data.sel(time=timestamp, method="nearest"))

        if save:
            date = pd.to_datetime(timestamp).strftime("%Y%m%d%H%M%S")
            htmlfile = self.filepath.replace(".pkl", f"_{date}.html")
            map.save(htmlfile)
            print(f"Map saved as {htmlfile}")

        return map


    def find_attached_datacubes(
            self,
            mode : Optional[str] = 'graph'
    ):
        """
        Find and return data cubes attached to the current instance based on the specified mode.

        In `graph` mode, it provides a graphical representation of the workflow based on the
        attached data cubes. In `dict` mode, it returns a dictionary of attached data cubes
        with detailed file paths and status. The method identifies attached data cubes by
        searching for files in the specified directory that match the current instance name
        and have a `.pkl` extension.

        Parameters:
        ----------
        mode : str, optional
            The mode in which to return the attached data cubes. Possible values are:
            'graph' - attempts to return a graphical representation (default).
            'dict' - returns a dictionary of attached data cubes.

        Returns:
        ----------
        dict or None
            In 'dict' mode, it returns a dictionary where keys are data cube statuses and
            values are dictionaries with the file path and a boolean indicating if it is
            the current data cube. In 'graph' mode, it attempts to return a workflow graph
            representation of the attached data cubes, or falls back to 'dict' mode.
        """

        attached = {}

        files = os.listdir(self.path)
        for file in files:
            if file.startswith(self.name) and file.endswith(".pkl"):

                status = find_status(file)
                if status != self.status:
                    attached[status] = {"filepath": os.path.join(self.path, file), "current": False}

        attached[self.status] = {"filepath": self.filepath, "current": True}

        if mode == 'graph':
            try:
                from DeThermIn.data.workflow_graph import get_workflow_graph
                return get_workflow_graph(attached)
            except:
                mode = 'dict'

        if mode == 'dict':
            return attached

        else:
            raise Exception("mode must be 'graph' or 'dict'.")


    def save(self):
        """
        Saves the current state of the data to a file.
        Format is .pkl, that can be opened using the pickles module.
        """

        if not os.path.exists(os.path.dirname(self.filepath)):
            os.makedirs(os.path.dirname(self.filepath))

        pickle.dump(self.data, open(self.filepath, "wb"))

        return


    def fusion(
            self,
            temporal_datacube,
            inplace : Optional[bool] = False
    ):
        """
        Fuse a DataCube with the current DataCube.

        This method combines a temporal datacube with the current datacube to create
        a fused datacube. The operation ensures spatial alignment by clipping and
        reprojecting the temporal datacube to match the spatial properties of the
        current datacube. The method also updates the status of the resultant datacube
        and applies compression and saving operations if necessary.

        Parameters:
        ----------
        temporal_datacube : DataCube
            The datacube to be fused.
            This parameter represents the temporal datacube that will be
            combined with the current datacube.
        inplace: bool, optional
            Whether to perform the operation in-place. Defaults to False.

        Returns:
        ----------
        DataCube
            The resulting fusioned datacube.
        """

        if self.temporal_resolution is not None:
            raise Exception("DataCube already has a temporal axis.")

        dc = inplace_rule(self, inplace)

        temporal_datacube.data = temporal_datacube.data.rio.clip_box(*dc.data.rio.bounds(), crs=dc.data.rio.crs)
        temporal_datacube.data = temporal_datacube.data.rio.reproject_match(dc.data, resampling=Resampling.bilinear)

        dc.data = xr.merge([dc.data, temporal_datacube.data])

        dc.update_status("FUSIONED")

        dc.compress()
        dc.save()

        return dc


    def enrich(
            self,
            verbose : Optional[bool] = True,
            inplace : Optional[bool] = False
    ):
        """
        Enriches a DataCube object with additional layers and features.

        This method applies various enrichment operations to a DataCube object, including
        calculations of HSV channels, distance grids,
        illumination properties, irradiance loading, and shadow rebuilding. The enrichment
        is only applicable to DataCubes in the "FUSIONED" status. Once enriched, the DataCube
        status is updated to "ENRICHED".

        Parameters:
        ----------
        verbose : bool, optional
            Whether to print progress information during the enrichment process.
        inplace : bool, optional
            Whether to perform the operation in-place. Defaults to False.

        Returns:
        ----------
        DataCube
            The enriched DataCube object if `inplace` is set to False; otherwise,
            the current DataCube is modified and returned.
        """

        if self.enriched:
            raise Exception("DataCube is already enriched.")

        if self.status != "FUSIONED": # to reactivate later
            raise Exception(f"Wrong status ({self.status}). You can only use `enrich` for a FUSIONED datacube (try to use `fusion` first)")

        dc = inplace_rule(self, inplace)

        if verbose:
            print("Getting extra layers (HSV channels, distance grids)...")
        dc.data = get_distance_grids(dc.data)
        dc.data = get_hsv_channels(dc.data)

        dc.data = get_illumination(dc.data, verbose=verbose)

        dc.data = get_irradiance_loading(dc.data, time_window=4, frequency=0.25, verbose=verbose)

        dc.data = rebuild_shadow(dc.data, verbose=verbose)

        dc.data = get_irradiance(dc.data, verbose=verbose)

        dc.update_status("ENRICHED")

        dc.compress()
        dc.save()

        return dc


    def to_train_datacube(
            self,
            ground_truth_dataset : xr.Dataset,
            inplace : Optional[bool] = True
    ):
        """
        Converts the current datacube to a training-ready datacube.

        This method merges the provided ground truth dataset with the existing data
        in the current datacube, updating its status to indicate readiness for
        training.

        Parameters:
        ----------
        ground_truth_dataset : xr.Dataset
            The dataset containing ground truth data to be merged.
        inplace : bool, optional
            Whether to perform the operation in-place. Defaults to True.

        Returns:
        ----------
        Datacube
            The datacube with the ground truth dataset merged and its status updated.
        """

        dc = inplace_rule(self, inplace)

        dc.data = xr.merge([ground_truth_dataset, dc.data])

        dc.update_status("TRAININGREADY")

        return dc


    def train(self):
        """
        Trains a Super-Resolution model using the current datacube, which must have the TRAININGREADY status.
        """

        if self.status != "TRAININGREADY":
            raise Exception(f"Wrong status ({self.status}). You can only use `train` for a TRAININGREADY datacube.")

        from DeThermIn.workflows import SR_trainer  # import here to avoid circular imports
        SR_trainer.train(self.filepath)

        return


def read_datacube(filepath : str) -> DataCube:
    """
    Reads a serialized datacube from the given file path.

    This function deserializes a datacube file and returns it as a DataCube object.
    A FileNotFoundError is raised if the file does not exist in the specified filepath.
    An Exception is raised if the file is not a .pkl file.

    Parameters:
    ----------
    filepath : str
        The path to the file containing the serialized datacube.

    Returns:
    ----------
    DataCube
        The deserialized DataCube object.
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError()

    elif not filepath.endswith(".pkl"):
        raise Exception("File must be a .pkl file.")

    dataset = pickle.load(open(filepath, "rb"))

    datacube = DataCube(dataset, filepath)

    return datacube


def inplace_rule(self: DataCube, inplace: bool) -> DataCube:
    """
    Determines whether to operate on the current instance of DataCube or
    create a new copy, based on the value of the 'inplace' parameter.

    Parameters:
    ----------
    inplace: bool
        If True, the operation will be performed on the current
        instance. If False, a new copy of the DataCube will be created
        and returned.

    Returns:
    ----------
    DataCube
        The current or copied DataCube instance, depending on the
        value of the 'inplace' parameter.
    """

    if not inplace:
        dc = DataCube(
            data=deepcopy(self.data),
            filepath=self.filepath,
        )
    else:
        dc = self

    return dc