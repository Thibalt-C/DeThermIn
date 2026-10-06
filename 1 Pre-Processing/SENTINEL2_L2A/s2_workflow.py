import s2_tools as s2
import pickle
import sys
import os
import openeo
import time

gdf_pkl = sys.argv[1]
start = sys.argv[2]
end = sys.argv[3]
name = sys.argv[4]
pickle_files_folder = sys.argv[5]
size = float(sys.argv[6])
extension = sys.argv[7]
folder = sys.argv[8]; folder = os.path.join(folder, "multispectral/")

gdf = pickle.load(open(gdf_pkl, 'rb'))

filepath = os.path.join("data", name + ".tif")

if not os.path.exists(filepath):

    start_time = time.time()
    max_time = 3600 # s
    success = False

    while not success:
        try:

            connection = s2.connection

            result = s2.prepare_request(connection, gdf, start, end, size)

            filepath = s2.request_and_download(result, name, folder)

            success = True

        except openeo.rest.OpenEoApiPlainError:
            success = False
            time.sleep(10)

            new_time = time.time()
            if new_time-start_time >= max_time:
                raise RuntimeError()

dataset = s2.create_s2_dataset(filepath)

dataset = s2.get_ndvi(dataset)

dataset = s2.get_ndwi(dataset)

dataset = s2.get_endisi(dataset)

pickle.dump(dataset, open(f"{pickle_files_folder}{name}_{extension}.pkl", "wb"))

print("-> multispectral workflow is done.")