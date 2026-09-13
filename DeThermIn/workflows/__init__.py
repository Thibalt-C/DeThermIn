'''
WORKFLOWS MODULE
Description: Initializes the workflows module linked to the DeThermIn workflow. Manages the different workflows associated to the DeThermIn project.
Author: Thibault Chardon
Creation date: 2026-08-16
'''

from DeThermIn.workflows.SR_trainer import *
from DeThermIn.workflows.workflow_manager import *

__all__ = [
    "pre_process",
    "spatio_temporal_fusion",
    "super_resolution",
    "DeThermIn_workflow",
    "get_training_dataset"
]