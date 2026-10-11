from fusion_tools.DTC.DTC_model.DTC_params import get_LST_params, get_extra_time_params
from fusion_tools.DTC.DTC_model.DTC_model import DTC_equation
import numpy as np
import matplotlib.pyplot as plt
import sys

path = sys.argv[1]
processed_data_path = sys.argv[2]
name = sys.argv[3]
extension = sys.argv[4]

t = np.linspace(0, 24, 240)

tsun = 14
tzen = 9
lst_sunrise = 10
lst_zen = 38
lst_next = 15


omega, u = get_extra_time_params(tsun, tzen)
T0, Ta, omega, delta_T, k = get_LST_params(lst_zen, lst_sunrise, lst_next, tsun, omega, u)
params = [tzen, tsun, T0, Ta, omega, delta_T, k]
LST = DTC_equation(t, *params)


fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(t, LST, c="k")

ax.set_xlabel("Time past sunrise [h]", fontsize=14)
ax.set_ylabel("LST [°C]", fontsize=14)

ax.tick_params(labelsize=12)

ax2 = ax.twiny()
ax2.set_xticks([0,tzen, tsun, t[-1]],[r"$t_{☉\!\uparrow}$",r"$t_{☉max}$",r"$t_{☉\!\downarrow}$", r"$t_{☉\!\uparrow\!\! +1}$"], fontsize=15)
ax3 = ax2.twinx()
ax3.set_yticks([lst_sunrise, lst_zen, T0, LST[-1]], [r"$T(t_{☉\!\uparrow})$", r"$T(t_{☉max})$", r"$T_0$", r"$T(t_{☉\!\uparrow\!\!+1})$"], fontsize=15)
ax2.grid(True)
ax3.grid(True)


T_T_0_lower = t[t<tzen][abs(LST[t<tzen] - T0).argmin()]
T_T_0_upper = t[t>tzen][abs(LST[t>tzen] - T0).argmin()]

ax3.annotate('', xy=(T_T_0_lower, T0), xytext=(T_T_0_upper, T0), arrowprops=dict(arrowstyle='<->', color='black', lw=1.5) )
ax3.text((T_T_0_lower+T_T_0_upper)/2, T0, r"$\omega$", va="top", ha="center", fontsize=15)

ax2.annotate('', xy=(T_T_0_upper, T0), xytext=(T_T_0_upper, LST.min()), arrowprops=dict(arrowstyle='<->', color='black', lw=1.5) )
ax3.text(T_T_0_upper, (T0+LST.min())/2, r"  $\delta T$", va="center", ha="left", fontsize=15)

ax2.annotate('', xy=((T_T_0_lower+T_T_0_upper)/2, T0), xytext=((T_T_0_lower+T_T_0_upper)/2, LST.max()), arrowprops=dict(arrowstyle='<->', color='black', lw=1.5) )
ax3.text((T_T_0_lower+T_T_0_upper)/2, (T0+LST.max())/2, r"  $T_a$", va="center", ha="left", fontsize=15)

ax.set_xlim([0,t[-1]])
ax2.set_xlim([0,t[-1]])
ax.set_ylim([10,40])
ax3.set_ylim([10,40])

fig.savefig(path+"DTC_model.png", bbox_inches="tight", dpi=300)