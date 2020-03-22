import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, number_of_features, hl1, hl2):
        super(SimpleNet, self).__init__()

        self.fc1 = nn.Linear(number_of_features, hl1)
        self.fc2 = nn.Linear(hl1, hl2)
        self.fc3 = nn.Linear(hl2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x