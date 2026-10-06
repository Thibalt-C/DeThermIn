import pickle
import os
import sys
from tqdm import tqdm
import xarray as xr

model_path = "trained_models/"

def main():
    dataset_path = sys.argv[1]
    mode = sys.argv[2].lower()
    enriched_ext = sys.argv[3]
    end_product_ext = sys.argv[4]
    training_ready_ext = sys.argv[5]

    model_filepath = f"{model_path}ResUNet_{mode}_model.pkl"

    if os.path.exists(model_filepath):
        model = pickle.load(open(model_filepath, "rb"))
    else:
        raise FileNotFoundError(f"Model file not found: {model_filepath}. Please train the model first.")

    dataset = pickle.load(open(dataset_path, "rb"))

    iterator = tqdm(range(len(dataset.time.values)), desc=f"building DeThermIn product (mode={mode})")

    preds = []
    for t in iterator:
        pred = model.predict(dataset.isel(time=t))
        pred = xr.DataArray(
            pred,
            dims=("y", "x"),
            coords={"y": dataset.y, "x": dataset.x, "time": dataset.isel(time=t).time}
        )
        preds.append(pred)

    preds = xr.concat(preds, dim="time")
    preds = preds.rio.write_crs(dataset.rio.crs)
    preds.attrs = {"plot_kwargs": {"cmap":"RdYlBu_r"},
                   "standard_name": "Super-Resolution Land Surface Temperature",
                   "units": "°C"}

    dataset["SR_LST"] = preds
    SR_dataset_path = dataset_path.replace(enriched_ext, end_product_ext).replace(training_ready_ext, end_product_ext)

    pickle.dump(dataset, open(SR_dataset_path, "wb"))


if __name__ == "__main__":
    main()
