from ..workflows import get_training_datacube
from ..metadata import resolution, training_features, training_features_path
from .utils import bold_text

def train_ResUNet():
    """
    Script allowing the user to (re-)train the ResUNet model.
    """

    print(f"{bold_text("Warning")}: the model will be trained on different sets of features defined in: {training_features_path}")
    print("\nHere are the current sets of features defined:")
    for key in training_features.keys():
        print(f"{bold_text(key)}: {str(training_features[key])}")
    input("\nPress Enter to begin the training...")
    print("\n\n")

    training_dc = get_training_datacube(resolution=resolution)
    training_dc.train()

    return