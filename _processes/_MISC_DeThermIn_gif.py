from DeThermIn.data import read_datacube
from DeThermIn.metadata import parameters, extensions
import sys

path = sys.argv[1]

filepath = parameters["processed_data_path"] + "2_3573_48_9222_06_01_2026_08_31_2026" + "_" + extensions["end_product"] + ".pkl"

dc = read_datacube(filepath)

dc.plot("SR_LST", gridlines=False, savefig=path+"DeThermIn_example.gif")