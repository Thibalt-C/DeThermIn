import torch
import numpy as np

def get_positions(patch_size, stride, H, W):
    positions_i = list(range(0, H - patch_size + 1, stride))
    if positions_i[-1] != H - patch_size:
        positions_i.append(H - patch_size)

    positions_j = list(range(0, W - patch_size + 1, stride))
    if positions_j[-1] != W - patch_size:
        positions_j.append(W - patch_size)
    return positions_i, positions_j

def extract_patches(X, y, patch_size=64, stride=32):
    H, W, C = X.shape
    patches_X, patches_y, valid_patches = [], [], []

    positions_i, positions_j = get_positions(patch_size, stride, H, W)

    for i in positions_i:
        for j in positions_j:
            Xi = X[i:i+patch_size, j:j+patch_size, :]
            yi = y[i:i+patch_size, j:j+patch_size, :]

            if np.isnan(yi).any() or np.isnan(Xi).any():
                valid_patches.append(False)
            else:
                valid_patches.append(True)
                patches_X.append(Xi)
                patches_y.append(yi)

    patches_X = np.stack(patches_X)
    patches_y = np.stack(patches_y)

    return patches_X, patches_y, valid_patches


def pyramid_weight(patch_size):
    # 1D ramp up then down
    ramp = np.linspace(0, 1, patch_size // 2)
    ramp = np.concatenate([ramp, ramp[::-1]])

    # 2D mask
    w = np.outer(ramp, ramp)
    return w.astype(np.float32)

def reconstruct_from_patches(patches, original_shape, valid_patches,
                             patch_size=64, stride=32):

    H, W = original_shape
    recon = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    positions_i, positions_j = get_positions(patch_size, stride, H, W)

    pw = pyramid_weight(patch_size)

    idx = 0
    k = 0

    for i in positions_i:
        for j in positions_j:

            if valid_patches[k]:
                patch = patches[idx, 0].cpu().numpy()

                recon[i:i+patch_size, j:j+patch_size] += patch * pw
                weight[i:i+patch_size, j:j+patch_size] += pw

                idx += 1

            k += 1

    mask = weight != 0
    recon[mask] /= weight[mask]
    recon[~mask] = np.nan

    return recon


def reconstruct_DL(model, loader, patch_size, stride, H, W, valid_patches):
    predictions = []

    with torch.no_grad():
        for X_batch, _ in loader:
            out = model(X_batch)
            predictions.append(out)

    predictions = torch.cat(predictions, dim=0)

    reconstructed = reconstruct_from_patches(predictions, (H, W), valid_patches = valid_patches,
                                             patch_size=patch_size, stride=stride)

    return reconstructed


