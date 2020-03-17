import torch
import torch.utils.data

class BuildingDataset(Dataset):
    def __init__(self, X_filepath, y_filepath):
        self.inputs = torch.from_numpy(np.load(X_filepath)).float()
        self.targets = torch.

