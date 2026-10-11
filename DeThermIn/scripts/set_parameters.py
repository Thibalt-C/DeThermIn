from .utils import bold_text
from ..metadata import (parameters,
                        parameters_description,
                        json_filepath_parameters,
                        update_lidar_metadata
                        )
import json

def set_params():
    """
    Script to edit the pàrameters of the workflow.
    """

    print("\n\n---------- PARAMETERS ----------\n")

    for key in parameters.keys():
        param = parameters[key]
        dtype, description = parameters_description[key]
        print(f"{bold_text(key)} ({dtype}): \n{description} \n-> {param}\n")

    print("\n")

    answer = ""
    while answer.lower() != "q":

        answer = input(f"To edit a parameter, first type the name of the parameter to edit. Press [{bold_text("Q")}] to exit. ")

        if answer.lower() in parameters.keys():

            parameters[answer] = input(f"Please enter the new value for {answer}: ")

            with open(json_filepath_parameters, "w") as f:
                json.dump(parameters, f, indent=4)

            if answer.lower() == "downloaded_data_path":
                update_lidar_metadata(parameters["downloaded_data_path"])
            elif answer.lower() == "processed_data_path":
                update_data_blending_path(parameters["processed_data_path"])

        else:
            print("Parameter not found. Please try again.")

    print("\nNew parameters:")
    print("--------------------")

    for key in parameters.keys():
        param = parameters[key]
        print(f"{bold_text(key)}: {param}")

    return