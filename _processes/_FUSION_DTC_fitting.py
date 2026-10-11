import matplotlib.pyplot as plt
import datetime
import matplotlib.gridspec as gridspec
import matplotlib
import matplotlib.colors as colors
import numpy as np
import pickle
from fusion_tools import import_dataset
from fusion_tools.DTC import split_in_cycles
from fusion_tools.metrics import rmse
import sys
import os

path = sys.argv[1]
processed_data_path = sys.argv[2]
name = sys.argv[3]
extension = sys.argv[5]

dtc_fitted_path = os.path.join(processed_data_path, f"{name}_{extension}.pkl")
dtc_fitted_dataset = import_dataset(dtc_fitted_path)

cycles = split_in_cycles(dtc_fitted_dataset)

fig, ax = plt.subplots(figsize=(10, 3))
dtc_fitted_dataset.LST_rebuilt.isel(x=6,y=3).plot(c='r')
dtc_fitted_dataset.LST.isel(x=6,y=3).plot.scatter(c='k', x='time', marker="+")
ax.set_ylabel("Land Surface Temperature [°C]")
ax.set_xlabel("")
ax.set_title("")
ax.set_xticks([cycles[key].sunrise for key in cycles.keys()],
              [datetime.datetime.strftime(cycles[key].sunrise, "%Y-%m-%d") + r" $t_{☉\!\uparrow}$" for key in cycles.keys()]
              , fontsize=12)
ax.grid(True)
ax.set_xlim([datetime.datetime(2026, 6, 24, 4),
             datetime.datetime(2026, 6, 28, 5)])
ax.set_ylim([15, 50])
plt.tight_layout()
fig.savefig(path+"fitted_DTC_model_example.png", bbox_inches="tight", dpi=300)




x = dtc_fitted_dataset.LST
y = dtc_fitted_dataset.LST_rebuilt
residuals = x - y
resid_std = np.nanstd(residuals.values)

bias = np.nanmean(residuals.values)
RMSE = rmse(x.values, y.values)


fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])  # top row taller

ax_0 = fig.add_subplot(gs[0, 0])   # top-left
ax_1 = fig.add_subplot(gs[0, 1])   # top-right
ax_2 = fig.add_subplot(gs[1, :])   # bottom row spans both columns

axes = [ax_0, ax_1, ax_2]

ax2 = axes[0].twiny()

axes[0].hist(residuals.values.flatten(),
         bins=np.linspace(-4*resid_std,4*resid_std,200),
         color="k")

ax2.set_xticks([-4*resid_std, -3*resid_std, -2*resid_std, -resid_std, 0, resid_std, 2*resid_std, 3*resid_std, 4*resid_std],
               [r"$-4\sigma$", r"$-3\sigma$", r"$-2\sigma$", r"$-\sigma$", 0 , r"$\sigma$", r"$2\sigma$", r"$3\sigma$", r"$4\sigma$"])

axes[0].set_xlabel("Residuals [°C]", fontsize=14)
axes[0].set_ylabel("# of pixels", fontsize=14)

axes[0].legend(title=f"RMSE: {RMSE:.2f} °C\nbias: {bias:.2f} °C", title_fontsize=14)


norm = colors.Normalize(vmin=3, vmax=24)
cmap = matplotlib.colormaps["jet"]

for key in cycles.keys():
    cycle = cycles[key]
    i = 7
    j = 4
    LST = cycle.isel(y=i,x=j).LST_rebuilt.values
    true = cycle.isel(y=i,x=j).LST.values
    resid = LST - true
    count = len(true[np.isfinite(true)])
    if count > 0:
        color = cmap(norm(count))
        axes[1].scatter(LST, true, color=color,marker=".",alpha=0.5)


sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes[1])
cbar.set_label("# of available observations per cycle", fontsize=12, labelpad=-10)
cbar.set_ticks([3,24])
cbar.ax.tick_params(labelsize=12)

axes[1].plot(np.linspace(0,50,3),np.linspace(0,50,3),"--k")
axes[1].set_xlim([0,50])
axes[1].set_ylim([0,50])

axes[1].set_xlabel("DTC_model-fitted geostationary LST [°C]", fontsize=14)
axes[1].set_ylabel("Observed geostationary LST [°C]", fontsize=14)


cycle_bins_residuals = residuals.groupby_bins("cycle_time", np.linspace(0, 24, 25))

data = [group.values.flatten()[np.isfinite(group.values.flatten())] for _, group in cycle_bins_residuals]

axes[2].boxplot(data, positions=np.arange(24), widths=0.6, showfliers=False)
axes[2].set_xlabel("Time of the day since sunrise [h]", fontsize=14)
axes[2].set_ylabel("Residuals [°C]", fontsize=14)
axes[2].grid(True)
axes[2].plot(np.linspace(-1,24,2), np.zeros(2), "--k")
axes[2].set_xlim([-1,24])
axes[2].set_ylim([-6,6])
plt.tight_layout()

fig.savefig(path+"DTC_ultra_coarse_residuals.png", bbox_inches="tight", dpi=300)