import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, number_of_features):
        super(SimpleNet, self).__init__()

        self.fc1 = nn.Linear(number_of_features, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x