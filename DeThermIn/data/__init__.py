'''
DATA MODULE
Description: Initializes the data module linked to the DeThermIn workflow. Handles the objects containing the data of interest.
Author: Thibault Chardon
Creation date: 2026-08-16
'''

from DeThermIn.data.datacube import *
from DeThermIn.data import *

__all__ = [
    "DataCube", # class object for data
    "read_datacube" # pkl file reader to DataCube
]
