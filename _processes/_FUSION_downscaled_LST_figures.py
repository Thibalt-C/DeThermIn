from fusion_tools import import_dataset
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import matplotlib.patheffects as path_effects

path = sys.argv[1]
processed_data_path = sys.argv[2]
name = sys.argv[3]
extension = sys.argv[4]
extension2 = sys.argv[6]

def add_scale(ax, dataset, scale_size=200, div_factor=20):
    resol = dataset.rio.resolution()[0]
    n_pixels = int(scale_size//resol)
    x = dataset.x
    y = dataset.y
    center_x = len(x)//div_factor
    lower_y = len(y)//div_factor
    y_idxs = [-lower_y , -lower_y]
    x_idxs = [center_x, center_x+n_pixels]
    line = ax.plot(x[x_idxs], y[y_idxs], "-", color="w", linewidth=10)
    text = ax.text(x[x_idxs[1]], y[y_idxs[1]], f"   {scale_size} m", ha="left", va="center", fontweight="bold", color="w", fontsize=12)
    text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground="black"),
            path_effects.Normal()
        ])
    line[0].set_path_effects([
            path_effects.Stroke(linewidth=13, foreground="black"),
            path_effects.Normal()
        ])
    return

fusioned_path = os.path.join(processed_data_path, f"{name}_{extension}.pkl")
fusioned_dataset = import_dataset(fusioned_path)

fig, axes = plt.subplots(2,4, figsize=(14, 4.5))

ims = []
layers = []

vmin = fusioned_dataset.fine_LST.min().values
vmax = fusioned_dataset.fine_LST.max().values

for i in range(4):
    idx = i * 6 + 1
    layer_coarse = fusioned_dataset.isel(time=idx).coarse_LST
    layer_fine = fusioned_dataset.isel(time=idx).fine_LST
    im1 = axes[0,i].imshow(layer_coarse, cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
    im2 = axes[1,i].imshow(layer_fine, cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
    axes[0,i].set_title(
        pd.to_datetime(fusioned_dataset.time.values[idx]).strftime("%Y-%m-%d %H:%M UTC"),
        fontsize=14)

    ims.extend([im1,im2])
    layers.extend([layer_coarse, layer_fine])

fig.subplots_adjust(
    hspace=0.05,
    wspace=0.05
)

cax = fig.add_axes([0.1, 0.05, 0.83, 0.04])
cbar = fig.colorbar(ims[-1], cax=cax, orientation="horizontal")
cbar.set_ticks([vmin, vmax])
cbar.set_label("Land Surface Temperature [°C]", fontsize=14, labelpad=-10)

for ax in axes:
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

axes[0,0].set_ylabel("LST @3,000 m", fontsize=14)
axes[1,0].set_ylabel("LST @70 m", fontsize=14)

#add_scale(axes[1,0], fusioned_dataset, scale_size=2000)

fig.savefig(path+"downscaled_LST_results.png", bbox_inches="tight", dpi=300)




narrow_swath_path = os.path.join(processed_data_path, f"{name}_{extension2}.pkl")
narrow_swath_dataset = import_dataset(narrow_swath_path)

narrow_swath_dataset = narrow_swath_dataset.rio.reproject_match(fusioned_dataset)

NDVI = narrow_swath_dataset.NDVI
MNDWI = narrow_swath_dataset.MNDWI
ENDISI = narrow_swath_dataset.ENDISI
water_mask = MNDWI-ENDISI>0.5
high_endisi_mask = ENDISI>0
high_ndvi_mask = (NDVI>0.75) & (ENDISI<0)


fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey="row")

for mask, color, n in zip([water_mask, high_ndvi_mask, high_endisi_mask], ["b", "g", "r"], range(3)):

    masked_fusioned = fusioned_dataset.fine_LST.where(mask)
    masked_raw = narrow_swath_dataset.LST.where(mask)

    median = masked_raw.median(("x","y"))
    q1 = masked_raw.quantile(0.25,("x","y"))
    q3 = masked_raw.quantile(0.75,("x","y"))

    median.plot.scatter(color=color, x="time", ax=axes[0,n], label="median observed \nLST @70 m")
    q1.plot.scatter(marker="v", color=color, x="time", ax=axes[0,n], label="1st quantile \nobserved LST @70 m")
    q3.plot.scatter(marker="^", color=color, x="time", ax=axes[0,n], label="3rd quantile \nobserved LST @70 m")

    masked_fusioned.quantile(0.25,("x","y")).plot(color="k", linestyle="--", ax=axes[0,n], label="1st/3rd quantile \nsynthetic LST @70 m")
    masked_fusioned.median(("x","y")).plot(color="k", ax=axes[0,n], label="median \nsynthetic LST @70 m")
    masked_fusioned.quantile(0.75,("x","y")).plot(color="k", linestyle="--", ax=axes[0,n])

axes[1,0].imshow(water_mask, cmap="Blues")
axes[1,1].imshow(high_ndvi_mask, cmap="Greens")
axes[1,2].imshow(high_endisi_mask, cmap="Reds")

for ax in axes[1]:
    ax.axis("off")

for ax in axes[0]:
    if ax==axes[0,0]:
        ax.set_ylabel("Land Surface Temperature [°C]", fontsize=13)
    else:
        ax.set_ylabel("")
    ax.legend(ncol=2, loc="upper right", fontsize=10.5)
    ax.set_xlabel("")
    date0 = pd.to_datetime(masked_fusioned.time.values[0]).date()
    ax.grid(True)
    ax.set_xlim([masked_fusioned.time[0], masked_fusioned.time[-1]])

axes[0,0].set_title("NDWI-ENDISI > 0.5", fontsize=18)
axes[0,1].set_title("NDVI > 0.75 & ENDISI < 0", fontsize=18)
axes[0,2].set_title("ENDISI > 0", fontsize=18)

fig.subplots_adjust(wspace=0.05, hspace=0.05)

plt.tight_layout()
fig.savefig(path+"downscaled_LST_land_cover_comparison", bbox_inches="tight", dpi=300)
