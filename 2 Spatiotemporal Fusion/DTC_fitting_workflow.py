from fusion_tools import import_dataset, fit_DTC
import pickle
import sys
import os

geostationary_path = sys.argv[1]
alpha_1 = float(sys.argv[2])
alpha_2 = float(sys.argv[3])
tau = float(sys.argv[4])
geostationary_ext = sys.argv[5]
dtc_fitted_ext = sys.argv[6]

DTC_fitted_path = geostationary_path.replace(geostationary_ext, dtc_fitted_ext)
geostationary_dataset = import_dataset(geostationary_path)

if os.path.exists(DTC_fitted_path):
    rebuilt = pickle.load(open(DTC_fitted_path, "rb"))

else:
    rebuilt = fit_DTC(geostationary_dataset,
                      alpha_monotonicity=alpha_1,
                      alpha_zenith=alpha_2,
                      sunset_shift=tau
                      )
    pickle.dump(rebuilt, open(DTC_fitted_path, "wb"))

print("-> Fitted a Diurnal Temperature Cycle model model to the geostationary data.")