import numpy as np
import os
import matplotlib.pyplot as plt
import sys

path = sys.argv[1]


store = {}

ECOSTRESS_bands = [1.660, 8.285, 8.785, 9.060, 10.522, 12.001]

fig, ax = plt.subplots(figsize=(14, 3))

colors = ["red", "orange", "green", "cyan", "blue", "magenta", "purple"]

i=0

for file in os.listdir("_processes/ecospeclib-1779026895242"):

    if file.__contains__("spectrum"):

        filepath = os.path.join("_processes/ecospeclib-1779026895242", file)

        with open(filepath, "r") as f:
            first_line = f.readline()
            name = first_line.split(": ")[-1].split("\n")[0]

        spectral_data = np.loadtxt(filepath, skiprows=21)

        store[name] = spectral_data

        ax.plot(spectral_data[:, 0], spectral_data[:, 1], label=name, color=colors[i])

        i+=1

ax.set_xlabel("Wavelength [µm]", fontsize=12)
ax.set_ylabel("Reflectance [%]", fontsize=12)
ax.set_xlim([2,14])
ax.set_ylim([0,70])
for band in ECOSTRESS_bands:
    ax.axvline(band, ls=":", c="k", lw=2, label=f"ECOSTRESS TIR bands" if band==ECOSTRESS_bands[0] else None)
ax.legend(fontsize=12, ncols=2, loc="upper right")
fig.savefig(path+"ECOSTRESS_library_surface_reflectances.png", dpi=300, bbox_inches="tight")


T_max = 273.15 + 100
T_med = 273.15 + 20
T_min = 273.15 - 20
wl = np.linspace(1, 20, 100) # µm
h = 6.62607015 * 1e-34
k = 1.380649 * 1e-23
c = 299792458

fig, ax = plt.subplots(figsize=(14, 2.2))
for T, color in zip([T_max, T_med, T_min], ["red", "orange", "blue"]):
    lmda = wl * 1e-6
    radiance = 2 * h * c**2 / lmda**5 * 1/(np.exp(h * c / (k * T * lmda)) - 1)  * 1e-6 # W/m^2/sr/um
    ax.plot(wl, radiance, label=f"T = {T-273.15} °C, black-body", ls="--", c=color)

    lmda = store["Alluvial sand"][:, 0] * 1e-6
    rad = 2 * h * c ** 2 / lmda ** 5 * 1 / (np.exp(h * c / (k * T * lmda)) - 1) * (1 - store["Alluvial sand"][:, 1] / 100) * 1e-6 # W/m^2/sr/um
    ax.plot(store["Alluvial sand"][:, 0], rad, label=f"T = {T - 273.15} °C, {"Alluvial sand"}", c=color)

for band in ECOSTRESS_bands:
    ax.axvline(band, ls=":", c="k", lw=2, label=f"ECOSTRESS TIR bands" if band==ECOSTRESS_bands[0] else None)

ax.legend()
ax.set_xlim([1, 20])
ax.set_xlabel("Wavelength [µm]", fontsize=12)
ax.set_ylabel(r"Spectral radiance " "\n" r"[W/m$^2$/sr/µm]", fontsize=12)
fig.savefig(path+"ECOSTRESS_library_spectral_radiances.png", dpi=300, bbox_inches="tight")
