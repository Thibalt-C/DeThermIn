'''
DeThermIn INITIALIZATION SCRIPT
Description: Initializes the DeThermIn module.
Author: Thibault Chardon
Creation date: 2026-05-01
'''

from .data import DataCube, read_datacube
from .workflows import *
from .scripts import selection, set_params, train_ResUNet, predownload, download_available_areas
from .metadata import parameters