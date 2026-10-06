"""
This scripts aims to acquire from the user the parameters enabling to select the area and date of interest.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from ..data.datacube import read_datacube
from ..metadata import extensions, parameters, start, end, resolution, bbox_size, training_features
from ..workflows import pre_process, DTC_fitting, spatio_temporal_fusion, super_resolution
from .utils import bold_text, get_processed_areas, select_processed_area, select_area_of_interest, check_requirements


def selection():
    """
    Script allowing the user to select an area of interest and run the DeThermIn workflow.
    """

    print(bold_text(
          "\n\n---------- Dense Thermal Information (DeThermIn) workflow ----------" +
          "\nauthor: Thibault CHARDON" +
          "\nhttps://github.com/Thibalt-C/DeThermIn" +
          "\nDelft University of Technology, The Netherlands, 2026" +
          "\n--------------------------------------------------------------------\n")
    )


    # 1. Selection of an area of interest

    processed_areas, filenames, regions, zones = get_processed_areas()
    processing_status, filename, from_scratch, region, zone = select_processed_area(processed_areas, filenames, regions, zones)

    narrow_swath_path = os.path.join(parameters["processed_data_path"], f"{filename}_{extensions["narrow-swath"]}.pkl")
    geostationary_path = os.path.join(parameters["processed_data_path"], f"{filename}_{extensions['geostationary']}.pkl")
    lidar_path = os.path.join(parameters["processed_data_path"], f"{filename}_{extensions['lidar']}.pkl")
    dtc_fitted_path = os.path.join(parameters["processed_data_path"], f"{filename}_{extensions["DTC-fitted"]}.pkl")
    fusioned_path = os.path.join(parameters["processed_data_path"], f"{filename}_{extensions['fusioned_LSTs']}.pkl")
    ancillary_path = os.path.join(parameters["processed_data_path"], f"{filename}_{extensions['blended_ancillary_data']}.pkl")
    DeThermIn_path = os.path.join(parameters["processed_data_path"], f"{filename}_{extensions['end_product']}.pkl")

    if from_scratch:

        lat, lon, city, region, country, size, AOI = select_area_of_interest()

        print("\n\nReady for the pre-processing step with the following parameters:")
        print(
            f" -> {bold_text("area of interest")}: square of size {size} m centered at {lon}°E, {lat}°N ({city}, {country})")
        print(f" -> {bold_text("resolution")}: {resolution} m")
        print(f" -> {bold_text("date range")}: {start.strftime("%Y-%m-%d")} to {end.strftime("%Y-%m-%d")}")
        print(f" -> {bold_text("larger bbox size")} (used for LST and multispectral data): {bbox_size} km")
        print("\n-------------------------------------------------")
        print(f"{bold_text("Warning")}: Ensure you have a stable connection to internet.")
        print(
            f"         Downloaded data will be stored in {parameters["processed_data_path"]}. Ensure this folder exists and has enough storage.")
        print(
            f"         You need to have pre-downloaded the geostationary data for this time range (use `predownload` command).")
        print(
            f"         This step can take a while especially for a very high resolution combined with a large area of interest.")
        print(
            f"         Status updates from the different operations done in parallel might be mixed up, but this won't affect the results.")
        print("-------------------------------------------------")
        input("\nPress Enter to start the pre-processing step...")


    # 2. Pre-processing of the data

    if processing_status == "partially pre-processed (0/4)":

        check_requirements()

        for file in os.listdir(parameters["geostationary_data_path"]):
            if file.__contains__(f"{region}"):
                geos_path = os.path.join(parameters["geostationary_data_path"], file)
        if not os.path.exists(geos_path):
            raise Exception("The geostationary data has not been downloaded. Please use the `predownload` command first.")

        if not from_scratch:
            dc = read_datacube(lidar_path)
            AOI = dc.get_gdf()
            AOI["zone"] = zone
            AOI["country"] = region
            start_ = datetime.datetime.strptime(dc.filepath.split("/")[-1][15:25], "%m_%d_%Y")
            end_ = datetime.datetime.strptime(dc.filepath.split("/")[-1][26:36], "%m_%d_%Y")
            resolution_ = dc.data.rio.resolution()[0]
            params = [AOI, start_, end_, resolution_, bbox_size, geos_path]
        else:
            params = [AOI, start, end, resolution, bbox_size, geos_path]

        ancillary_dc, narrow_swath_dc, geostationary_dc = pre_process(*params)

        processing_status = "pre-processed (1/4)"


    # 3. Fitting of a DTC model

    if processing_status == "pre-processed (1/4)":

        if not from_scratch:
            narrow_swath_dc = read_datacube(narrow_swath_path)
            geostationary_dc = read_datacube(geostationary_path)

        print("\nThe geostationary data needs to be fitted with a Diurnal Temperature Cycle model. Hyper-parameters are:")
        print(f" -> {bold_text("alpha_1")} (monotonicity regularisation): {parameters['DTC_alpha_1']}")
        print(f" -> {bold_text("alpha_2")} (maximimum regularisation): {parameters['DTC_alpha_2']}")
        print(f" -> {bold_text("tau")} (sunset shift): {parameters['DTC_sunset_shift']}")
        input("\nPress Enter to start the fitting...")

        dtc_fitted_dc = DTC_fitting(geostationary_dc)

        processing_status = "DTC_model-fitted (2/4)"


    # 4. Spatio-temporal fusion

    if processing_status == "DTC_model-fitted (2/4)":

        if not from_scratch:
            narrow_swath_dc = read_datacube(narrow_swath_path)
            dtc_fitted_dc = read_datacube(dtc_fitted_path)

        print("\nProceeding to spatio-temporal unmixing with the following parameters:")
        print(f" -> {bold_text('alpha_space')}: {parameters['fusion_alpha_space']}")
        print(f" -> {bold_text('alpha_time')}: {parameters['fusion_alpha_time']}")
        print(f" -> {bold_text('k')}: {parameters['fusion_k']}")
        print(f" -> {bold_text('window_size')}: {parameters['fusion_window_size']}")
        input("\nPress Enter to proceed to the day selection...")

        fusioned_dc = spatio_temporal_fusion(narrow_swath_dc, dtc_fitted_dc)

        processing_status = "fusioned (3/4)"


    # 5. Super-resolution

    if processing_status == "fusioned (3/4)":

        if not from_scratch:
            fusioned_dc = read_datacube(fusioned_path)
            ancillary_dc = read_datacube(ancillary_path)

        print(f"\n Here are the available super-resolution models (you may need to train them beforehand):")
        for key in training_features.keys():
            print(f"- {bold_text(key)}")
        mode = input("\nSelect a super-resolution model: ")

        DeThermIn_dc = super_resolution(ancillary_dc, fusioned_dc, mode)

        processing_status = "processed (4/4)"


    # 6. Visualise end-product

    if processing_status == "processed (4/4)":

        if not from_scratch:
            DeThermIn_dc = read_datacube(DeThermIn_path)

        sel = "m"

        while sel.lower() != "q":

            if sel.lower() == "m":
                print("\nChoose an option:")
                print(f"[{bold_text("I")}] make interactive map (html file) for a given timestamp")
                print(f"[{bold_text("D")}] generate a dynamical plot (gif file) of a given temporal feature")
                print(f"[{bold_text("P")}] plot a static feature")
                print(f"[{bold_text("Q")}] QUIT")
                sel = input("\n")

            elif sel.lower() == "i":

                timestamps = [pd.to_datetime(dt) for dt in DeThermIn_dc.data.time.values]
                for n,timestamp in enumerate(timestamps):
                    print(f"[{bold_text(str(n))}] {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                t = input("Select a timestamp: ")
                timestamp = timestamps[int(t)]
                DeThermIn_dc.make_interactive_map(timestamp)
                sel = "m"

            elif sel.lower() == "d":
                layers=[]; n=0
                for layer in DeThermIn_dc.data.data_vars:
                    if DeThermIn_dc.data.data_vars[layer].dims == ("time", "y", "x"):
                        try:
                            name = DeThermIn_dc.data.data_vars[layer].attrs['standard_name']
                            units = DeThermIn_dc.data.data_vars[layer].attrs['units']
                            print(f"[{bold_text(str(n))}] {name} ({units})")
                            layers.append(layer)
                            n+=1
                        except:
                            pass
                l = input("Select a layer: ")
                layer = layers[int(l)]
                DeThermIn_dc.plot(layer)
                plt.show()
                sel = "m"

            elif sel.lower() == "p":
                layers = []; n=0
                for layer in DeThermIn_dc.data.data_vars:
                    if DeThermIn_dc.data.data_vars[layer].dims == ("y", "x"):
                        try:
                            name = DeThermIn_dc.data.data_vars[layer].attrs['standard_name']
                            units = DeThermIn_dc.data.data_vars[layer].attrs['units']
                            print(f"[{bold_text(str(n))}] {name} ({units})")
                            layers.append(layer)
                            n += 1
                        except:
                            pass
                l = input("Select a layer: ")
                layer = layers[int(l)]
                DeThermIn_dc.plot(layer)
                plt.show()
                sel = "m"

            else:
                print("Invalid option.")
                sel = "m"



        if sel.lower() == "q":
            sys.exit()






