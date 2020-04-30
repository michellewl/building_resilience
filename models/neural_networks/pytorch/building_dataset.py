# This script defines an instance of a Dataset class used by the PyTorch machine learning library.
# This definition was written specifically for the ASHRAE building energy dataset but is applicable in other contexts.
# It is assumed that the inputs will be numpy files.

import torch
from torch.utils.data import Dataset
import numpy as np

class BuildingDataset(Dataset):
    r"""Arguments:
            X_filepath: File path to the inputs data as a .npy file.
            y_filepath: File path to the targets data as a .npy file.
        """

    def __init__(self, X_filepath, y_filepath):
        self.inputs = torch.from_numpy(np.load(X_filepath)).float()
        self.targets = torch.from_numpy(np.load(y_filepath)).float()
    # This currently loads the whole data file, but could be altered and moved to the get item function to define loading of a single row at a time.

    def __len__(self):
        return self.inputs.size()[0]

    def nfeatures(self):
        return self.inputs.size()[1]

    def __getitem__(self, index):
        input = self.inputs[index]
        target = self.targets[index]
        return {"inputs":input, "targets":target}

