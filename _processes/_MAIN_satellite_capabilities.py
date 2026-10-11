import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import sys

path = sys.argv[1]

plot_workflows = True


SLSTR = {"name": "SLSTR", "satellite": "S3", "time_res": 0.5, "spatial_res": 1000, "future": False, "marker": "p", "color": "blue"}
ECOSTRESS = {"name": "ECOSTRESS", "satellite": "ISS", "time_res": 1, "spatial_res": 70, "future": False, "marker": "D", "color": "green"}
FCI = {"name": "FCI", "satellite": "MTG", "time_res": 1/144, "spatial_res": 2000, "future": False, "marker": "h", "color": "blue"}
SEVERI = {"name": "SEVERI", "satellite": "MSG", "time_res": 1/96, "spatial_res": 3000, "future": False, "marker": "h", "color": "red"}
LANDSAT7 = {"name": "ETM+", "satellite": "L7", "time_res": 16, "spatial_res": 60, "future": False, "marker": "D", "color": "blue"}
LANDSAT8 = {"name": "TIRS", "satellite": "L8", "time_res": 16, "spatial_res": 30, "future": False, "marker": "D", "color": "purple"}
MODIS = {"name": "MODIS", "satellite": "Terra", "time_res": 1, "spatial_res": 1000, "future": False, "marker": "p", "color": "red"}
ASTER = {"name": "ASTER", "satellite": "Terra", "time_res": 16, "spatial_res": 90, "future": False, "marker": "D", "color": "red"}
LSTM = {"name": "LSTM", "satellite": "S8", "time_res": 2, "spatial_res": 50, "future": True, "launch_year":2028, "marker": "D", "color": "orange"}
SBG = {"name": "OTTER", "satellite": "SBG-TIR", "time_res": 3, "spatial_res": 60, "future": True, "launch_year":2029, "marker": "D", "color": "cyan"}
TRISHNA = {"name": "TIR", "satellite": "TRISHNA", "time_res": 3, "spatial_res": 50, "future": True, "launch_year":2027, "marker": "D", "color": "magenta"}
GOES = {"name": "ABI", "satellite": "GOES", "time_res": 1/24, "spatial_res": 2000, "future": False, "marker": "h", "color": "green"}


satellites = [FCI, SEVERI, GOES, SLSTR, MODIS, ECOSTRESS, LANDSAT7, LANDSAT8, ASTER, LSTM, SBG, TRISHNA]

existing_workflows = {

    "DST-UNET: Beber et al. (2025)" : [{"init_spatial_res": LANDSAT8["spatial_res"],
                                        "init_time_res": LANDSAT8["time_res"],
                                        "final_spatial_res": 0.5,
                                        "final_time_res": LANDSAT8["time_res"]}],

    "Herrero-Huerta et al. (2019)" : [{"init_spatial_res": LANDSAT8["spatial_res"],
                                       "init_time_res": LANDSAT8["time_res"],
                                       "final_spatial_res": LANDSAT8["spatial_res"],
                                       "final_time_res": MODIS["time_res"]},
                                      {"init_spatial_res": MODIS["spatial_res"],
                                       "init_time_res": MODIS["time_res"],
                                       "final_spatial_res": LANDSAT8["spatial_res"],
                                       "final_time_res": MODIS["time_res"]},
                                      {"init_spatial_res": LANDSAT8["spatial_res"],
                                       "init_time_res": MODIS["time_res"],
                                       "final_spatial_res": 10,
                                       "final_time_res": MODIS["time_res"]}
                                      ],

    "TsHARP: Huryna et al. (2019)" : [{"init_spatial_res": SLSTR["spatial_res"],
                                       "init_time_res": SLSTR["time_res"],
                                       "final_spatial_res": 60,
                                       "final_time_res": SLSTR["time_res"]}],

    "Bechtel et al. (2012)" : [{"init_spatial_res": SEVERI["spatial_res"],
                                "init_time_res": SEVERI["time_res"],
                                "final_spatial_res": 100,
                                 "final_time_res": SEVERI["time_res"]}],

    "Wen et al. (2022)" : [{"init_spatial_res": ECOSTRESS["spatial_res"],
                            "init_time_res": ECOSTRESS["time_res"],
                            "final_spatial_res": ECOSTRESS["spatial_res"],
                            "final_time_res": GOES["time_res"]},
                            {"init_spatial_res": GOES["spatial_res"],
                            "init_time_res": GOES["time_res"],
                            "final_spatial_res": ECOSTRESS["spatial_res"],
                            "final_time_res": GOES["time_res"]}
                           ],

    # "Proposed VHR DeThermIn workflow" : [{"init_spatial_res": FCI["spatial_res"],
    #                      "init_time_res": FCI["time_res"],
    #                      "final_spatial_res": SLSTR["spatial_res"],
    #                      "final_time_res": FCI["time_res"]},
    #                     #{"init_spatial_res": SLSTR["spatial_res"],
    #                      #"init_time_res": SLSTR["time_res"],
    #                      #"final_spatial_res": SLSTR["spatial_res"],
    #                      #"final_time_res": FCI["time_res"]},
    #                     {"init_spatial_res": SLSTR["spatial_res"],
    #                      "init_time_res": FCI["time_res"],
    #                      "final_spatial_res": ECOSTRESS["spatial_res"],
    #                      "final_time_res": FCI["time_res"]},
    #                     {"init_spatial_res": ECOSTRESS["spatial_res"],
    #                      "init_time_res": ECOSTRESS["time_res"],
    #                      "final_spatial_res": ECOSTRESS["spatial_res"],
    #                      "final_time_res": FCI["time_res"]},
    #                     {"init_spatial_res": ECOSTRESS["spatial_res"],
    #                      "init_time_res": FCI["time_res"],
    #                      "final_spatial_res": 0.5,
    #                      "final_time_res": FCI["time_res"]}
    #                     ],
#
# "Proposed HR workflow" : [
#                         {"init_spatial_res": SLSTR["spatial_res"],
#                          "init_time_res": SLSTR["time_res"],
#                          "final_spatial_res": ECOSTRESS["spatial_res"],
#                          "final_time_res": SLSTR["time_res"]},
#                         {"init_spatial_res": ECOSTRESS["spatial_res"],
#                          "init_time_res": ECOSTRESS["time_res"],
#                          "final_spatial_res": ECOSTRESS["spatial_res"],
#                          "final_time_res": SLSTR["time_res"]},
#                         {"init_spatial_res": ECOSTRESS["spatial_res"],
#                          "init_time_res": SLSTR["time_res"],
#                          "final_spatial_res": 0.5,
#                          "final_time_res": SLSTR["time_res"]}
#                         ],

}


times = []
spatials = []
proxies = []

fig, ax = plt.subplots(figsize=(26, 14))
plt.grid(True)
ax.set_xscale('log')
ax.set_yscale('log')

if plot_workflows:

    for key, color, ls in zip(existing_workflows.keys(), ["red", "blue", "orange", "magenta", "green", "grey"], ["-", "-", "-", "-","-", ":"]):

        workflows = existing_workflows[key]

        for workflow in workflows:

            ax.annotate(
                text="",
                xy=(workflow["final_time_res"], workflow["final_spatial_res"]),  # End point (B)
                xytext=(workflow["init_time_res"], workflow["init_spatial_res"]),  # Start point (A)
                arrowprops=dict(arrowstyle="->", color=color, lw=4, ls=ls, mutation_scale=30),
                label=key
            )
            ax.scatter(workflow["final_time_res"], workflow["final_spatial_res"], marker="o", c=color)

            if workflow["final_spatial_res"]!=100: # avoiding overlapping
                spatials.append(workflow["final_spatial_res"])

        proxies.append(Line2D([0], [0], color=color, linestyle=ls, label=key, linewidth=4))

for satellite in satellites:
    time = satellite["time_res"]
    space = satellite["spatial_res"]
    if not satellite["future"]:
        name = f'  {satellite["satellite"]}-{satellite["name"]}  '
    else:
        name = f'  {satellite["satellite"]}-{satellite["name"]} ({satellite["launch_year"]})  '


    times.append(time)

    spatials.append(space)

    proxies.append(ax.scatter(time, space,
                              marker=satellite["marker"], c=satellite["color"],
                              s= 250 if satellite["marker"]=="D" else 500,
                              zorder=10, label=name, edgecolors="k", linewidths=2))
    line = False

    if line:
        time = times[-2]
    else:
        time = times[-1]

    space = spatials[-1]

    if time>15:
        ha = "left"
    else:
        ha = "left"

    if (satellite["satellite"] == "TRISHNA") | (satellite["satellite"] == "Landsat-7"):
        va = "top"
    else:
        va = "center"
    if (satellite["satellite"] == "S3") | (satellite["satellite"] == "S8")  :
        ha = "right"

fs = 28
ax.set_xlabel("Temporal resolution", fontsize=fs)
ax.set_ylabel("Spatial resolution [m]", fontsize=fs)
ax.legend(handles=proxies, fontsize=20, ncol=3)

times_ticks = []
for i in range(len(times)):
    t = times[i]
    unit = "d"
    if t < 1:
        t *= 24
        unit = "h"
        if t < 1:
            t *= 60
            unit = "min"
    times_ticks.append(f"{t:.0f} {unit}")

spatials = np.unique(spatials)
spatials.sort()
sp = np.array([str(int(spatial)) if spatial>=1 else str(spatial)  for spatial in spatials])
sp[spatials==60] = ""
sp[spatials==90] = ""

ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yscale(ax.get_yscale())

fs = 18
ax2.set_yticks([60,90], [60,90], fontsize=fs)
ax.set_xticks(times, times_ticks, fontsize=fs)
ax.set_yticks(spatials, sp, fontsize=fs)

if plot_workflows:
    plt.savefig(path+"satellite_capabilities.png", bbox_inches='tight', dpi=300)
else:
    plt.savefig(path+"satellite_capabilities_light.png", bbox_inches='tight', dpi=300)