import subprocess
import os
import pickle
import datetime
from DeThermIn.workflows import data_blending
from DeThermIn.data import read_datacube
from DeThermIn.workflows.workflow_manager import pre_process, spatio_temporal_fusion
from DeThermIn.workflows.metadata import metadata

ground_truth_path = metadata["ground_truth_path"]
ground_truth_script = metadata["ground_truth_script"]
ground_truth_pickles = metadata["ground_truth_pickles"]
sr_training_script = metadata["sr_training_script"]
sr_training_path = metadata["sr_training_path"]


def get_training_dataset(resolution, # meters
                         framing_interval=150, # days
                         training_set_name=""
                         ):

    process = subprocess.Popen(["python", ground_truth_script],
                               cwd=os.path.join(os.path.curdir, ground_truth_path),
                               stdout=subprocess.PIPE,
                               universal_newlines=True,
                               bufsize=1,
                               env=os.environ.copy())

    for line in iter(process.stdout.readline, ''):
        print(line, end='')

    process.wait()

    files = os.listdir(ground_truth_pickles)
    gdf_files = []
    for file in files:
        if file.__contains__('_gdf'):
            gdf_files.append(file)
            files.remove(file)

    ground_truths = {}

    for gdf_file in gdf_files:
        name = gdf_file.split('_gdf')[0]
        ground_truths[name] = {"gdf": gdf_file}

    for file in files:
        name = file.split('.pkl')[0]
        ground_truths[name]["dataset"] = file

    for key in ground_truths.keys():
        gdf = pickle.load(open(ground_truth_pickles + ground_truths[key]["gdf"], "rb"))
        ground_truths[key]["coords"] = gdf.union_all().centroid.coords[0]
        ground_truths[key]["datetime"] = list(gdf["datetime"])
        ground_truths[key]["geodataframe"] = gdf

    if training_set_name=="":
        print("Available ground truths:")
        for key in ground_truths.keys():
            print(f"- {key}: {ground_truths[key]['coords'][1]:.4f}°N {ground_truths[key]['coords'][0]:.4f}°E, {ground_truths[key]['datetime'][0].strftime('%d/%m/%Y %H:%M')} to {ground_truths[key]['datetime'][-1].strftime('%d/%m/%Y %H:%M')}")

        selection = ""
        while selection not in ground_truths.keys():
            selection = input("Type the name of the ground truth you want to use.")
    else:
        selection = training_set_name

    ground_truth = ground_truths[selection]
    gdf = ground_truth["geodataframe"]

    if ground_truth["datetime"][0]==ground_truth["datetime"][-1]:
        start = ground_truth["datetime"][0] - datetime.timedelta(days=framing_interval//2)
        end = ground_truth["datetime"][-1] + datetime.timedelta(days=framing_interval//2)
    else:
        start = ground_truth["datetime"][0]
        end = ground_truth["datetime"][-1]

    start = start.strftime('%m-%d-%Y')
    end = end.strftime('%m-%d-%Y')

    print(f"Selected {selection}: {start} to {end}.")

    gt_dataset = pickle.load(open(ground_truth_pickles + ground_truth["dataset"], "rb"))

    blended_dc, lsr_htr_dc, msr_ltr_dc, vlsr_vhtr_dc = pre_process(gdf, start, end, resolution)

    small_start = ground_truth["datetime"][0].strftime("%Y-%m-%d")
    small_end = ground_truth["datetime"][-1].strftime("%Y-%m-%d")

    fusioned_dc = spatio_temporal_fusion(blended_dc, lsr_htr_dc, msr_ltr_dc, vlsr_vhtr_dc, small_start, small_end)

    fusioned_dc.enrich(inplace=True)

    gt_dataset = gt_dataset.rio.reproject_match(fusioned_dc.data)

    dc = fusioned_dc.to_train_datacube(gt_dataset)

    dc.compress(inplace=True)
    dc.save()

    return dc


def train(training_dataset_path):

    process = subprocess.Popen(["python", sr_training_script, training_dataset_path],
                               cwd=os.path.join(os.path.curdir, sr_training_path),
                               stdout=subprocess.PIPE,
                               universal_newlines=True,
                               bufsize=1,
                               env=os.environ.copy())

    for line in iter(process.stdout.readline, ''):
        print(line, end='')

    process.wait()

    return