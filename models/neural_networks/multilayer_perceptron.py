# This script defines feed-forward neural networks using the PyTorch machine learning library.
# Each class instance has a different number of hidden layers, some with batch normalisation incorporated
# (this was added in as the models were developed, so all are included here; future work should use batch norm).

import torch
import torch.nn as nn

### 4 hidden layers with batch norm

class SimpleNet_4bn(nn.Module):
    def __init__(self, number_of_features, hl1, hl2, hl3, hl4):
        super(SimpleNet_4bn, self).__init__()

        self.fc1 = nn.Linear(number_of_features, hl1, bias=False)
        self.bn1 = nn.BatchNorm1d(hl1)
        self.fc2 = nn.Linear(hl1, hl2, bias=False)
        self.bn2 = nn.BatchNorm1d(hl2)
        self.fc3 = nn.Linear(hl2, hl3, bias=False)
        self.bn3 = nn.BatchNorm1d(hl3)
        self.fc4 = nn.Linear(hl3, hl4, bias=False)
        self.bn4 = nn.BatchNorm1d(hl4)
        self.fc5 = nn.Linear(hl4, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x

### 4 hidden layers; no batch norm

class SimpleNet_4(nn.Module):
    def __init__(self, number_of_features, hl1, hl2, hl3, hl4):
        super(SimpleNet_4, self).__init__()

        self.fc1 = nn.Linear(number_of_features, hl1)
        self.fc2 = nn.Linear(hl1, hl2)
        self.fc3 = nn.Linear(hl2, hl3)
        self.fc4 = nn.Linear(hl3, hl4)
        self.fc5 = nn.Linear(hl4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

### 3 hidden layers, no batch norm

class SimpleNet_3(nn.Module):
    def __init__(self, number_of_features, hl1, hl2, hl3):
        super(SimpleNet_3, self).__init__()

        self.fc1 = nn.Linear(number_of_features, hl1)
        self.fc2 = nn.Linear(hl1, hl2)
        self.fc3 = nn.Linear(hl2, hl3)
        self.fc4 = nn.Linear(hl3, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

### 3 hidden layers with batch norm

class SimpleNet_3bn(nn.Module):
    def __init__(self, number_of_features, hl1, hl2, hl3):
        super(SimpleNet_3bn, self).__init__()

        self.fc1 = nn.Linear(number_of_features, hl1, bias=False)
        self.bn1 = nn.BatchNorm1d(hl1)
        self.fc2 = nn.Linear(hl1, hl2, bias=False)
        self.bn2 = nn.BatchNorm1d(hl2)
        self.fc3 = nn.Linear(hl2, hl3, bias=False)
        self.bn3 = nn.BatchNorm1d(hl3)
        self.fc4 = nn.Linear(hl3, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

# 2 hidden layers; no batch norm

class SimpleNet_2(nn.Module):
    def __init__(self, number_of_features, hl1, hl2):
        super(SimpleNet_2, self).__init__()

        self.fc1 = nn.Linear(number_of_features, hl1)
        self.fc2 = nn.Linear(hl1, hl2)
        self.fc3 = nn.Linear(hl2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x