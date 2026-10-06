from super_resolution_tools import DL_regression
import pickle
import datetime
import os
import sys
import multiprocessing
from sr_workflow import model_path
import json


def main():

    dataset_path = sys.argv[1]
    features_file = sys.argv[2]
    train_size = float(sys.argv[3])
    valid_size = float(sys.argv[4])
    max_epochs = int(sys.argv[5])
    batch_size = int(sys.argv[6])
    patch_size = int(sys.argv[7])
    learning_rate = float(sys.argv[8])
    lambda_ssim = float(sys.argv[9])

    dataset = pickle.load(open(dataset_path, "rb"))

    sample = dataset.sel(
        time=datetime.datetime(2021, 6, 15, 11),
        method="nearest"
    )

    with open(features_file, "r") as f:
        features_lists = json.load(f)

    for key, features in features_lists.items():
        print(f"\nTraining for features set '{key}': {str(features)}")
        model , result = DL_regression(sample, features,
                                       train_size=train_size,
                                       valid_size=valid_size,
                                       epochs=max_epochs,
                                       batch_size=batch_size,
                                       patch_size=patch_size,
                                       learning_rate=learning_rate,
                                       lambda_ssim=lambda_ssim,
                                       plot=True, save_fig=True, filepath=f"results/{key}.png"
                                       )
        pickle.dump(model, open(os.path.join(model_path, f"ResUNet_{key}_model.pkl"), "wb"))

        for key in result.keys():
            result[key] = str(result[key])
        with open(f"results/{key}.json", "w") as f:
            json.dump(result, f, indent=4)

    print("\n-> Super-Resolution training is done.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
