import ecostress_tools as eco
import pickle
import sys
import os

import warnings
warnings.filterwarnings("ignore")

gdf_pkl = sys.argv[1]
start = sys.argv[2]
end = sys.argv[3]
name = sys.argv[4]
pickle_files_folder = sys.argv[5]
size = float(sys.argv[6])
extension = sys.argv[7]
username = sys.argv[8]
password = sys.argv[9]
folder = sys.argv[10]; folder = os.path.join(folder, "narrow_swath_LST/")

gdf = pickle.load(open(gdf_pkl, 'rb'))

filenames = eco.request_and_download(gdf, start, end, size, username, password, folder)

LST_dataset = eco.import_dataset(filenames)

LST_dataset = eco.drop_out_of_bounds_frames(LST_dataset, gdf)

pickle.dump(LST_dataset, open(f"{pickle_files_folder}{name}_{extension}.pkl", "wb"))

print("-> ECOSTRESS workflow is done.")