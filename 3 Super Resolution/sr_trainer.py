from super_resolution_tools import dl_regression
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
    #train_size = float(sys.argv[3])
    #valid_size = float(sys.argv[4])
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
        model_filepath = os.path.join(model_path, f"ResUNet_{key}_model.pkl")
        if not os.path.exists(model_filepath):

            print(f"\nTraining for features set '{key}': {str(features)}")

            model , result = dl_regression(sample, features,
                                           epochs=max_epochs,
                                           batch_size=batch_size,
                                           patch_size=patch_size,
                                           learning_rate=learning_rate,
                                           lambda_ssim=lambda_ssim,
                                           plot=True, filepath_base=f"results/{key}"
                                           )
            pickle.dump(model, open(os.path.join(model_path, f"ResUNet_{key}_model.pkl"), "wb"))

            for k in result.keys():
                # noinspection PyTypeChecker
                result[k] = str(result[k])
            with open(f"results/{key}.json", "w") as f:
                json.dump(result, f, indent=4)

        else:
            print(f"Model already exists for features set '{key}'. Skipping...")

    print("\n-> Super-Resolution training is done.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
