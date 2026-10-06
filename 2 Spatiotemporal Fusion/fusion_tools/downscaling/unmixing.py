import xarray as xr
import rioxarray
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from ..DTC.DTC_model import *
from ..metrics import *
from .matching import match_cycles, get_pairs, get_target
from tqdm import tqdm
from typing import Optional
from datetime import datetime

def estimator(
        img_window : np.ndarray,
        cycle_time : float,
        pairs_window : xr.Dataset,
        alpha_time : float,
        alpha_space : float,
        k : float,
        plot_weights : Optional[bool] = False
) -> float :

    synthetic_LST = 0

    # get a temporal weighting based on the relative time of the cycle
    relative_time = abs((pairs_window.cycle_time.values) - cycle_time)
    relative_time = np.array([float(np.nanmin([rt, 24 - rt])) for rt in relative_time])
    temporal_weights = np.exp(-alpha_time*relative_time)
    temporal_weights /= np.nansum(temporal_weights)

    ndvi = pairs_window.isel(time=0).NDVI.values
    ndwi = pairs_window.isel(time=0).MNDWI.values
    endisi = pairs_window.isel(time=0).ENDISI.values

    for t in range(len(pairs_window.time)):

        fine_img = pairs_window.isel(time=t).LST.values
        coarse_img = pairs_window.isel(time=t).coarse_LST.values

        X, Y = np.meshgrid(np.arange(fine_img.shape[0]), np.arange(fine_img.shape[1]))
        distances = np.sqrt((X - fine_img.shape[0] // 2) ** 2 + (Y - fine_img.shape[1] // 2) ** 2)

        # NDVI filtering
        sigma = np.nanstd(ndvi)
        central_pixel_ndvi = ndvi[fine_img.shape[0] // 2, fine_img.shape[1] // 2]
        lower_bound = central_pixel_ndvi - k * sigma
        upper_bound = central_pixel_ndvi + k * sigma
        similar_ndvi = (ndvi >= lower_bound) & (ndvi <= upper_bound)

        # NDWI filtering
        sigma = np.nanstd(ndwi)
        central_pixel_ndwi = ndwi[fine_img.shape[0] // 2, fine_img.shape[1] // 2]
        lower_bound = central_pixel_ndwi - k * sigma
        upper_bound = central_pixel_ndwi + k * sigma
        similar_ndwi = (ndwi >= lower_bound) & (ndwi <= upper_bound)

        # ENDISI filtering
        sigma = np.nanstd(endisi)
        central_pixel_endisi = endisi[fine_img.shape[0] // 2, fine_img.shape[1] // 2]
        lower_bound = central_pixel_endisi - k * sigma
        upper_bound = central_pixel_endisi + k * sigma
        similar_endisi = (endisi >= lower_bound) & (endisi <= upper_bound)

        # LST filtering
        sigma = np.nanstd(fine_img)
        central_pixel_LST = fine_img[fine_img.shape[0] // 2, fine_img.shape[1] // 2]
        lower_bound = central_pixel_LST - k * sigma
        upper_bound = central_pixel_LST + k * sigma
        similar_LST = (fine_img >= lower_bound) & (fine_img <= upper_bound)

        # Removing pixels that are not similar enough to the central pixel
        similar_pixels = similar_ndvi & similar_ndwi & similar_endisi & similar_LST

        # Computing weights based on the distance to the central pixel and similarity
        weights = np.exp(-alpha_space*distances)
        weights[~similar_pixels] = 0
        weights /= np.nansum(weights)

        if plot_weights:
            fig, axs = plt.subplots(1,4, figsize=(20, 4))
            im = axs[0].imshow(fine_img, cmap='RdYlBu_r')
            fig.colorbar(im, ax=axs[0])
            im = axs[1].imshow(coarse_img, cmap='RdYlBu_r')
            fig.colorbar(im, ax=axs[1])
            axs[2].imshow(similar_pixels, cmap='RdYlBu_r')
            im = axs[3].imshow(weights, cmap='plasma')
            fig.colorbar(im, ax=axs[3])
            axs[0].set_title("MSR LST")
            axs[1].set_title("LSR LST")
            axs[2].set_title("Similar pixels")
            axs[3].set_title("Weights")
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            plt.show()

        add = np.nansum(weights * (fine_img + img_window - coarse_img))

        synthetic_LST += add * temporal_weights[t]

    return synthetic_LST


def process_unmixing(
        coarse_target : np.ndarray,
        cycle_time : float,
        pairs : xr.Dataset,
        window_size : int,
        alpha_time : float,
        alpha_space : float,
        k : float
) -> xr.DataArray :


    def Convolve2d_parallel(image, window_size, cycle_time, pairs, alpha_time, alpha_space, k, func=estimator, n_jobs=4):
        h, w = image.shape
        out_h, out_w = h - window_size + 1, w - window_size + 1
        def process_row(i):
            row = np.empty(out_w)
            for j in range(out_w):
                pairs_window = pairs.isel(y=slice(i, i + window_size), x=slice(j, j + window_size))
                row[j] = func(image[i:i + window_size, j:j + window_size],
                              cycle_time, pairs_window, alpha_time, alpha_space, k)
            return row
        rows = Parallel(n_jobs=n_jobs, verbose=False)(delayed(process_row)(i) for i in tqdm(range(out_h)))
        return np.array(rows)

    synthetic_LST = Convolve2d_parallel(coarse_target, window_size, cycle_time, pairs, alpha_time, alpha_space, k)

    n = window_size-1

    synthetic_LST = xr.DataArray(synthetic_LST, dims=("y","x"),
                                 coords={"y": pairs.y[n//2:-n//2], "x": pairs.x[n//2:-n//2]})

    return synthetic_LST


def spatio_temporal_unmixing(
        coarse_dataset : xr.Dataset,
        fine_dataset : xr.Dataset,
        day : datetime,
        alpha_space : float,
        alpha_time : float,
        k : float,
        window_size : int
) -> xr.Dataset :

    print("Matching coarse and fine datasets...")
    matched = match_cycles(coarse_dataset, fine_dataset)

    pairs = get_pairs(matched)
    target = get_target(coarse_dataset, fine_dataset, day)

    # LST parameters to downscale
    coarse_LST_sunrise = target.isel(time=0).LST_sunrise.values
    coarse_LST_zenith = target.isel(time=0).LST_max.values
    coarse_LST_next_sunrise = target.isel(time=0).LST_next_sunrise.values

    cycle_time_sunrise = 0.0
    cycle_time_zenith = float(target.isel(time=0).thermal_zenith.mean())

    hyperparams = [window_size, alpha_time, alpha_space, k]


    print("Processing sunrise LST (1/3)")
    target["LST_sunrise"] = process_unmixing(coarse_LST_sunrise, cycle_time_sunrise, pairs, *hyperparams)

    print("Processing thermal zenith LST (2/3)")
    target["LST_max"] = process_unmixing(coarse_LST_zenith, cycle_time_zenith, pairs, *hyperparams)

    print("Processing next sunrise LST (3/3)")
    target["LST_next_sunrise"] = process_unmixing(coarse_LST_next_sunrise, cycle_time_sunrise, pairs, *hyperparams)

    fine_LST = compute_LST(target.cycle_time.values,
                           target.isel(time=0).thermal_sunset.values,
                           target.isel(time=0).thermal_zenith.values,
                           target.LST_max.values,
                           target.LST_sunrise.values,
                           target.LST_next_sunrise.values,
                           mode = "3D_array"
                           )

    target["fine_LST"] = xr.DataArray(fine_LST, dims=("time", "y", "x"),
                                      coords={"y": target.y,
                                              "x": target.x,
                                              "time": target.time
                                              },
                                      attrs={"units": "°C",
                                             "standard_name": "synthetic Land Surface Temperature",
                                             "plot_kwargs": {"cmap": "RdYlBu_r"}
                                             }
                                                )
    for var in target.data_vars:
        if var in ["LST", "LST_uncertainty", "sunset_time", "QFLAGS", "Q_FLAGS", "delta_LST0", "diurnal_cycle",
                   "thermal_sunset", "thermal_zenith", "LST_max", "LST_sunrise", "LST_next_sunrise"]:
            target = target.drop_vars(var)
    target = target.rename({"LST_rebuilt": "coarse_LST"})

    target = target.where(np.isfinite(target.fine_LST).any(axis=0), drop=True) # remove borders

    return target


# def leave_one_our_cross_validation(medium_res_dataset, rebuilt_cycles, cycles, estimators,
#                                    window_size, alpha, k, daytime=1):
#
#     medium_res_cycles, coarse_imgs, msr_imgs, indexes = prepare_for_unmixing(medium_res_dataset, rebuilt_cycles, cycles, estimators, daytime=daytime)
#
#     RMSEs = []
#
#     for j,leave_one_out in enumerate(indexes):
#
#         coarse_imgs_filtered = [element for i, element in enumerate(coarse_imgs) if i != leave_one_out]
#         msr_imgs_filtered = [element for i, element in enumerate(msr_imgs) if i != leave_one_out]
#
#         n = window_size - 1
#
#         target = xr.DataArray(coarse_imgs[j], dims=("y","x"),
#                               coords={"y":medium_res_cycles.y, "x":medium_res_cycles.x}).rio.write_crs(medium_res_cycles.rio.crs)
#         true = msr_imgs[j][n//2:-n//2,n//2:-n//2]
#
#         synthetic = process_unmixing(target, medium_res_cycles, coarse_imgs_filtered, msr_imgs_filtered,
#                                      window_size=window_size, alpha=alpha, k=k)
#
#         RMSE = rmse(synthetic.values, true)
#
#         print(RMSE)
#
#         synthetic.plot(cmap="RdYlBu_r")
#         plt.show()
#
#         RMSEs.append(RMSE)
#
#
#     return np.mean(RMSEs)
#
#
# def grid_search_cv_unmixing(medium_res_dataset, rebuilt_cycles, cycles, estimators, window_sizes, alphas, ks):
#
#     results = []
#
#     for alpha in alphas:
#         for k in ks:
#             for window_size in window_sizes:
#
#                 RMSE = leave_one_our_cross_validation(medium_res_dataset, rebuilt_cycles, cycles, estimators, window_size, alpha, k)
#                 results.append({"window_size": window_size, "alpha": alpha, "k": k, "RMSE": RMSE})
#
#     return results