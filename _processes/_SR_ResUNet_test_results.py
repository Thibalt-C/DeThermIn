import os
import pickle
from super_resolution_tools.plotter import plot_test_predictions
from super_resolution_tools.CNNs.patch_maker import extract_patches
import sys
import multiprocessing
import datetime
from sr_workflow import model_path

path = sys.argv[1]
processed_data_path = sys.argv[2]
name = sys.argv[3]
extension = sys.argv[4]

mode = "dynamic"

def main():

    dataset_path = os.path.join(processed_data_path, f"{name}_{extension}.pkl")
    dataset = pickle.load(open(dataset_path, "rb"))

    model = pickle.load(open(os.path.join(model_path, f"ResUNet_{mode}_model.pkl"), "rb"))

    test = dataset.where(
        dataset.test_mask, drop=True
    ).sel(time=datetime.datetime(2021,6,15, 11), method="nearest")

    pred = model.predict(test)

    plot_test_predictions(pred, test, filepath=path+"super_resolution_result.png")

if __name__ == "__main__":
    multiprocessing.freeze_support()  # optional but safe on macOS
    main()