import datetime
from fusion_tools import import_dataset, spatio_temporal_unmixing
import pickle
import sys
import os

narrow_swath_path = sys.argv[1]
DTC_fitted_path = sys.argv[2]
day = sys.argv[3]; day = datetime.datetime.strptime(day, "%Y-%m-%d")
narrow_swath_ext = sys.argv[4]
fusioned_ext = sys.argv[5]
window_size = int(sys.argv[6])
alpha_time = float(sys.argv[7])
alpha_space = float(sys.argv[8])
k = float(sys.argv[9])

fusioned_path = narrow_swath_path.replace(narrow_swath_ext, fusioned_ext)

if os.path.exists(fusioned_path):
    print(f"File already exists: {fusioned_path}")

else:

    print("-> Proceeding to spatio-temporal unmixing...")

    fine_dataset = import_dataset(narrow_swath_path)
    coarse_dataset = import_dataset(DTC_fitted_path)

    synthetic_LST = spatio_temporal_unmixing(coarse_dataset, fine_dataset, day,
                                             alpha_space, alpha_time, k, window_size
                                             )

    pickle.dump(synthetic_LST, open(fusioned_path, "wb"))

print("-> Spatio-temporal fusion workflow is done.")