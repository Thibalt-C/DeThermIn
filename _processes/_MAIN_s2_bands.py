import matplotlib.pyplot as plt
import sys

path = sys.argv[1]

fig, ax = plt.subplots(figsize=(1,5))

bands = {"B02": [490-65, 490, 490+65],
         "B03": [560-35, 560, 560+35],
         "B04": [665-30, 665, 665+30],
         "B08": [842-115, 842, 842+115],
         "B11": [1610-90, 1610, 1610+90],
         "B12": [2200-180, 2200, 2200+180]}

colors = ["blue", "green", "red", "firebrick", "darkred", "black"]

for n,key, color in zip(range(len(bands)),bands.keys(),colors):

    band = bands[key]

    plt.fill_betweenx(band, 0.25+n, 0.75+n, alpha=1, color=color, edgecolor=color)
    #plt.text(0.5+n, band[1], key, ha='center', va='center')

ax.set_xticks([])
ax.set_xlim([0,len(bands)+0.5])

y_ticks = [bands[key][1] for key in bands.keys()]
ax.set_yticks(y_ticks)
ax.set_ylim([400, 2500])

ax2 = ax.twinx()
ax2.set_yticks(y_ticks)
ax2.set_yticklabels(bands.keys())
ax2.set_ylim([400, 2500])
ax2.grid(True)

fig.savefig(path+'sentinel2_spectral_bands.png', bbox_inches='tight', dpi=300)