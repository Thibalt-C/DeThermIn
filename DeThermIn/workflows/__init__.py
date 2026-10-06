'''
WORKFLOWS MODULE
Description: Initializes the workflows module linked to the DeThermIn workflow.
Manages the different workflows associated to the DeThermIn project.
Author: Thibault Chardon
Creation date: 2026-08-16
'''

from .Trainer import train
from .workflow_manager import (pre_process,
                               DTC_fitting,
                               spatio_temporal_fusion,
                               super_resolution,
                               get_training_datacube)