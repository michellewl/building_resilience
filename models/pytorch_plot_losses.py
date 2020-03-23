from functions.functions import write, current_time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.building_dataset import BuildingDataset
import matplotlib.pyplot as plt
import numpy as np

windows_os = True

hidden_layer_1 = 400
hidden_layer_2 = 400
hidden_layer_3 = 400

arch = f"_{hidden_layer_1}_{hidden_layer_2}_{hidden_layer_3}"

if windows_os:
    code_home_folder = "C:\\Users\\Michelle\\OneDrive - University of Cambridge\\MRes\\Guided_Team_Challenge\\building_resilience\\"
    title = f"{code_home_folder}logs\\training\\daily_data\\MLP_pytorch_log_{current_time()}"
    data_folder = "data\\train_test_arrays\\"
    filename = f"{code_home_folder}models\\saved\\MLP_pytorch_model_daily{arch}.tar"
else:
    code_home_folder = "/home/mwlw3/Documents/Guided_team_challenge/building_resilience/"
    title = f"{code_home_folder}logs/training/daily_data/MLP_pytorch_log_{current_time()}"
    data_folder = "data/train_test_arrays/"
    filename = f"{code_home_folder}models/saved/MLP_pytorch_model_daily{arch}.tar"

training_losses = torch.load(filename)["training_loss_history"]
val_losses = torch.load(filename)["validation_loss_history"]

fig, ax = plt.subplots()
ax.plot(range(len(training_losses)), training_losses, label="training loss")
ax.plot(range(len(val_losses)), val_losses, label="validation loss")
plt.ylim(0, 1e-8)
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title(f"Hidden layer nodes: {hidden_layer_1}, {hidden_layer_2}, {hidden_layer_3}")
plt.show()