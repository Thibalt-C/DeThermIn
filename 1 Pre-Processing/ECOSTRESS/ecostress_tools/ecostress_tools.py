import datetime
import numpy as np
import xarray as xr
import rioxarray as rxr


def import_dataset(filenames):

    LST_files = []
    cloud_files = []
    quality_control_files = []

    for filename in filenames:
        file = filename.split("/")[-1]
        if file.__contains__("_LST_"):
            LST_files.append(filename)
        if file.__contains__("_cloud_mask_"):
            cloud_files.append(filename)
        if file.__contains__("_QC_"):
            quality_control_files.append(filename)


    def find_matching_date(timestamp, existing_dates, TOLERANCE=datetime.timedelta(seconds=30)):
        for d in existing_dates:
            if abs(d - timestamp) <= TOLERANCE:
                return d
        return None

    sets = {}
    for filename in LST_files:
        date = filename.split("_LST_")[-1][:15]
        date = datetime.datetime.strptime(date, "%Y%m%dT%H%M%S")
        sets[date] = {"LST_file": filename}
    for filename in cloud_files:
        date = filename.split("_cloud_mask_")[-1][:15]
        date = datetime.datetime.strptime(date, "%Y%m%dT%H%M%S")
        matching_date = find_matching_date(date, sets.keys())
        if matching_date is not None:
            sets[matching_date]["cloud_file"] = filename
    for filename in quality_control_files:
        date = filename.split("_QC_")[-1][:15]
        date = datetime.datetime.strptime(date, "%Y%m%dT%H%M%S")
        matching_date = find_matching_date(date, sets.keys())
        if matching_date is not None:
            sets[matching_date]["qc_file"] = filename

    sets = dict(sorted(sets.items()))
    time_axis = list(sets.keys())

    datasets = []

    for date in time_axis:

        LST_file = sets[date]["LST_file"]
        cloud_file = sets[date]["cloud_file"]
        qc_file = sets[date]["qc_file"]

        cloud_mask = rxr.open_rasterio(cloud_file, masked=True)
        cloud_attrs = cloud_mask.attrs
        cloud_mask = cloud_mask.squeeze()
        cloud_mask.attrs = cloud_attrs
        cloud_mask = cloud_mask.assign_attrs({"standard_name": "Cloud mask", "units": "no units", "plot_kwargs": {"cmap": "grey", "vmin":0, "vmax":1}})
        cloud_mask = cloud_mask.drop_vars("band")
        cloud_mask["time"] = date
        qc = rxr.open_rasterio(qc_file, masked=True)
        QC = qc.values.astype(np.int64)
        QC[QC == -99999] = -1
        qc.data = QC & 0b11
        qc_attrs = qc.attrs
        qc = qc.squeeze()
        qc.attrs = qc_attrs
        qc = qc.assign_attrs({"standard_name": "Quality control", "units": "no units", "plot_kwargs": {"cmap": "grey"}})
        qc = qc.drop_vars("band")
        qc["time"] = date
        qc = qc.astype(float)

        LST = rxr.open_rasterio(LST_file, masked=True)
        crs = LST.rio.crs
        LST_attrs = LST.attrs
        LST = LST.squeeze()
        LST.attrs = LST_attrs
        LST = LST.assign_attrs({"standard_name": "ISS-ECOSTRESS Land Surface Temperature",
                                "units": "°C",
                                "plot_kwargs": {"cmap": "RdYlBu_r"}})
        LST = LST.drop_vars("band")
        LST.values = LST.where(LST.values != 0, np.nan) * LST.attrs["scale_factor"] - 273.15 + LST.attrs["add_offset"]
        LST["time"] = date


        ecostress_dataset = xr.Dataset({"LST": LST, "cloud_mask": cloud_mask, "qc": qc})

        ecostress_dataset = ecostress_dataset.where(ecostress_dataset.cloud_mask != 1, np.nan)

        datasets.append(ecostress_dataset)

    reference_dataset = datasets[0]

    for ds in datasets[1:]:
        if (len(ds.x) > len(reference_dataset.x)) or (len(ds.y) > len(reference_dataset.y)):
            reference_dataset = ds

    aligned_datasets = []
    for ds in datasets:
        ds_aligned = ds.reindex_like(reference_dataset, method="nearest", tolerance=0.001)
        aligned_datasets.append(ds_aligned)

    temporal_LST_dataset = xr.concat(aligned_datasets, dim="time")

    temporal_LST_dataset.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    temporal_LST_dataset.rio.write_crs(crs, inplace=True)
    temporal_LST_dataset.rio.write_coordinate_system(inplace=True)

    return temporal_LST_dataset


def drop_out_of_bounds_frames(dataset, gdf):

    aoi = dataset.rio.clip(gdf.geometry, gdf.crs)

    for t in dataset.time.values:

        test = aoi.sel(time=t).LST.values
        if np.isnan(test).all():
            dataset = dataset.drop_sel(time=t)

    return dataset
