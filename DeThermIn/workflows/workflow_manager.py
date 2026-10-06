import datetime
import pandas as pd
from geopandas import GeoDataFrame
from typing import Tuple, Optional
import os

from . import PreProcessing as PP
from . import SpatioTemporalFusion as STF
from . import SuperResolution as SR
from . import Trainer as T

from ..data import read_datacube, DataCube
from ..metadata import bbox_size, extensions


def pre_process(
        gdf : GeoDataFrame,
        start : datetime.datetime,
        end : datetime.datetime,
        resolution : float,
        bbox_size : float,
        geostationary_path : str
) -> Tuple[DataCube, DataCube, DataCube] :
    """
    Launches the pre-processing workflows.

    Parameters:
    ----------
    gdf : `geopandas.GeoDataFrame`
        Geo-file of the area of interest
    start : `datetime.datetime`
        Lower time bound of the time interval considered
    end : `datetime.datetime`
        Higher time bound of the time interval considered
    resolution : float
        Spatial resolution of the ancillary data and end product, in meters
    bbox_size : float
        Size of the larger bounding box around the location of interest for LST downloads, in meters
    geostationary_path : string
        Path to the geostationary data

    Returns:
    ----------
    ancillary_dc : `DeThermIn.DataCube`
        Contains the ancillary data to be used later for the super-resolution workflow
    narrow_swath_dc : `DeThermIn.DataCube`
        Contains the LST and related data of the narrow swath mission (finer spatial resolution)
    geostationary_dc : `DeThermIn.DataCube`
        Contains the LST and related data of the geostationary mission (coarser spatial resolution)
    """
    geos_dc = read_datacube(geostationary_path)
    ancillary_dc = PP.launch_data_blending(gdf, start, end, resolution, bbox_size)
    datacubes = ancillary_dc.find_attached_datacubes(mode="dict")
    narrow_swath_dc = read_datacube(datacubes["narrow-swath"]["filepath"])
    geostationary_dc = PP.find_geostationary_data(geos_dc, narrow_swath_dc)

    return ancillary_dc, narrow_swath_dc, geostationary_dc


def DTC_fitting(geostationary_dc : DataCube) -> DataCube :
    """
    Launches the DTC fitting workflow.

    Parameters:
    ----------
    geostationary_dc : `DeThermIn.DataCube`
        Contains the LST and related data of the geostationary mission (coarser spatial resolution)

    Returns:
    ----------
    dtc_fitted_dc : `DeThermIn.DataCube`
        LST predicted by the DTC model and associated parameters, at the resolution of the input geostationary data
    """

    dtc_fitted_dc = STF.launch_DTC_fitting(geostationary_dc.filepath)

    return dtc_fitted_dc


def spatio_temporal_fusion(
        narrow_swath_dc : DataCube,
        dtc_fitted_dc : DataCube,
        day : datetime.datetime = None,
) -> DataCube :

    fusioned_dc = STF.launch_unmixing(narrow_swath_dc.filepath, dtc_fitted_dc.filepath, day)

    return fusioned_dc


def super_resolution(
        ancillary_dc : DataCube,
        fusioned_dc : DataCube,
        mode : Optional[str] = "dynamic"
) -> DataCube :

    enriched_dc_path = fusioned_dc.filepath.replace(extensions["fusioned_LSTs"],
                                                    extensions["enriched_ancillary_data"])

    if os.path.exists(enriched_dc_path):
        ancillary_dc = read_datacube(enriched_dc_path)

    else:

        blended_dc_path = fusioned_dc.filepath.replace(extensions["fusioned_LSTs"],
                                                       extensions["blended_ancillary_and_LST"])
        if os.path.exists(blended_dc_path):
            ancillary_dc = read_datacube(blended_dc_path)
        else:
            ancillary_dc.add_time_series(fusioned_dc, inplace=True) # blends ancillary data with LST time series

        ancillary_dc.enrich(inplace=True) # adds new ancillary features

    DeThermIn = SR.launch(ancillary_dc.filepath, mode=mode)

    return DeThermIn



def get_training_datacube(
        resolution : float,
        time_extent : Optional[int] = 90,
        bbox_size : Optional[float] = bbox_size
) -> DataCube :

    # generate ground truth
    ground_truth_dc = T.generate_ground_truth(resolution, time_extent)

    # pre-processing and DTC-fitting if necessary
    dtc_fitted_dc_path = ground_truth_dc.filepath.replace(extensions["ground_truth"], extensions["DTC-fitted"])
    narrow_swath_dc_path = ground_truth_dc.filepath.replace(extensions["ground_truth"], extensions["narrow-swath"])
    if not os.path.exists(dtc_fitted_dc_path):
        day = pd.to_datetime(ground_truth_dc.data.time.values)
        start = day - datetime.timedelta(days=time_extent//2)
        end = day + datetime.timedelta(days=time_extent//2)
        gdf = ground_truth_dc.get_gdf()
        geos_dc = T.download_geostationary_data(start, end, gdf)
        ancillary_dc, narrow_swath_dc, geostationary_dc = pre_process(gdf,
                                                                      start, end,
                                                                      resolution, bbox_size,
                                                                      geos_dc.filepath
                                                                      )
        dtc_fitted_dc = DTC_fitting(geostationary_dc)

    else:
        dtc_fitted_dc = read_datacube(dtc_fitted_dc_path)
        narrow_swath_dc = read_datacube(narrow_swath_dc_path)

    # spatio-temporal fusion if necessary
    fusioned_dc_path = dtc_fitted_dc.filepath.replace(extensions["DTC-fitted"], extensions["fusioned_LSTs"])
    if not os.path.exists(fusioned_dc_path):
        fusioned_dc = spatio_temporal_fusion(narrow_swath_dc, dtc_fitted_dc, day)
    else:
        fusioned_dc = read_datacube(fusioned_dc_path)

    # skipping blending if already done
    ancillary_dc_path = fusioned_dc.filepath.replace(extensions["fusioned_LSTs"], extensions["blended_ancillary_data"])
    blended_dc_path = fusioned_dc.filepath.replace(extensions["fusioned_LSTs"], extensions["blended_ancillary_and_LST"])
    if not os.path.exists(blended_dc_path):
        ancillary_dc = read_datacube(ancillary_dc_path)
        ancillary_dc.add_time_series(fusioned_dc, inplace=True)
    else:
        ancillary_dc = read_datacube(blended_dc_path)

    # enriching if necessary
    enriched_dc_path = fusioned_dc.filepath.replace(extensions["fusioned_LSTs"], extensions["enriched_ancillary_data"])
    if not os.path.exists(enriched_dc_path):
        ancillary_dc.enrich(inplace=True)
    else:
        ancillary_dc = read_datacube(enriched_dc_path)

    # anyway updating with the ground truth
    training_dc = ancillary_dc.to_train_datacube(ground_truth_dc)

    training_dc.compress(inplace=True)
    training_dc.save()

    return training_dc