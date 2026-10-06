import numpy as np
from pytorch_msssim import ms_ssim

def rmse(x,y):
    return np.sqrt(np.nansum((x-y)**2)/len(x))

def res_bias(x,y):
    return np.nanmean(y-x)

def mae(x,y):
    return np.nanmean(np.abs(x-y))

def psnr(x,y):
    return 20*np.log10(np.nanmax(x)/rmse(x,y))

def ssim(x, y, k1=0.01, k2=0.03, bits=32):

    max_val = (2 ** bits) - 1

    def norm(img):
        img = img.astype(np.float32)
        return max_val * (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img))

    x = norm(x)
    y = norm(y)

    C1 = (k1 * max_val)**2
    C2 = (k2 * max_val)**2

    mu_x = np.nanmean(x)
    mu_y = np.nanmean(y)
    sigma_x = np.nanvar(x)
    sigma_y = np.nanvar(y)
    sigma_xy = np.nanmean((x - mu_x) * (y - mu_y))

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    return num / den


def multi_scale_ssim(pred, target):
    return 1 - ms_ssim(pred, target, data_range=1, size_average=True, win_size=7)


def evaluate(y_pred_test, y_test, verbose=True):

    MAE = mae(y_test.squeeze(), y_pred_test)
    PSNR = psnr(y_test.squeeze(), y_pred_test)
    SSIM = ssim(y_test.squeeze(), y_pred_test)
    RMSE = rmse(y_test.squeeze(), y_pred_test)
    bias = res_bias(y_test.squeeze(), y_pred_test)

    if verbose:
        print(f"Final test RMSE: {RMSE:.2f} °C")
        print(f"Final test bias: {bias:.2f} °C")
        print(f"Final test MAE: {MAE:.2f} °C")
        print(f"Final test PSNR: {PSNR:.2f} dB")
        print(f"Final test SSIM: {100 * SSIM:.2f} %")

    return {"RMSE": RMSE, "bias": bias, "MAE": MAE, "PSNR": PSNR, "SSIM": SSIM}