import glob
import numpy as np
from functions.functions import write, current_time
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

from matplotlib import pyplot as plt

### Folder formatting ###

windows_os = True

if windows_os:
    code_home_folder = "C:\\Users\\Michelle\\OneDrive - University of Cambridge\\MRes\\Guided_Team_Challenge\\building_resilience\\"
    title = f"{code_home_folder}logs\\training\\daily_data\\MLP_pytorch_log_{current_time()}"
    data_folder = "data\\train_test_arrays\\"
    #filename = f"{code_home_folder}models\\MLP_model_daily{arch}.sav"
else:
    code_home_folder = "/home/mwlw3/Documents/Guided_team_challenge/building_resilience/"
    title = f"{code_home_folder}logs/training/daily_data/MLP_log_{current_time()}"
    data_folder = "data/train_test_arrays/"
    #filename = f"{code_home_folder}models/MLP_model_daily{arch}.sav"

### Code ###


# write(title, f"{current_time()}\nWEATHER TRAINING DATA\n")
# write(title, f"All buildings. Daily data, including meta data.")
#
# write(title, "\nBUILDING TRAINING DATA\n")
# write(title, f"All buildings, daily total energy.")

print("Importing data...")
#X_train = torch.from_numpy(np.load(glob.glob(f"{code_home_folder}{data_folder}X_train.npy")[0])).float()
#y_train = torch.from_numpy(np.load(glob.glob(f"{code_home_folder}{data_folder}y_train.npy")[0])).float()
X_test = torch.from_numpy(np.load(glob.glob(f"{code_home_folder}{data_folder}X_test.npy")[0])).float()
y_test = torch.from_numpy(np.load(glob.glob(f"{code_home_folder}{data_folder}y_test.npy")[0])).float()

# write(title, f"Training array dimensions: {X_train.shape} {y_train.shape}")
# write(title, f"Test array dimensions: {X_test.shape} {y_test.shape}")

print("Processing data...")

### Run with the smaller test set for now to develop code ###

batch_size = 16
num_epochs = 4
batches_per_print = 2000

tensor_X = X_test
tensor_y = y_test

print(tensor_X.size(), tensor_y.size())

dataset = TensorDataset(tensor_X, tensor_y)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.fc1 = nn.Linear(int(tensor_X.size()[1]), 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        # x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        #print(x.size())
        x = torch.relu(self.fc2(x))
        #print(x.size())
        x = self.fc3(x)
        #print(x.size())
        return x


simple_net = SimpleNet()

print(simple_net)

criterion = nn.MSELoss()
optimiser = optim.Adam(simple_net.parameters())



print("Begin training...")
for epoch in range(num_epochs):
    running_loss = 0.0

    for batch_num, data in enumerate(dataloader):
        inputs, targets = data


        optimiser.zero_grad()

        outputs = simple_net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimiser.step()

        # print statistics
        running_loss += loss.item()
        if batch_num % batches_per_print == batches_per_print-1:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, batch_num + 1, running_loss / batches_per_print))
            running_loss = 0.0

print('Finished Training')