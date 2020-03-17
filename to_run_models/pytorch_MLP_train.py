import glob
import numpy as np
from functions.functions import write, current_time
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from matplotlib import pyplot as plt

### Folder formatting ###

windows_os = True

if windows_os:
    code_home_folder = "C:\\Users\\Michelle\\OneDrive - University of Cambridge\\MRes\\Guided_Team_Challenge\\building_resilience\\"
    title = f"{code_home_folder}logs\\training\\daily_data\\MLP_pytorch_log_{current_time()}"
    data_folder = "data\\train_test_arrays\\"
    filename = f"{code_home_folder}models\\MLP_model_daily{arch}.sav"
else:
    code_home_folder = "/home/mwlw3/Documents/Guided_team_challenge/building_resilience/"
    title = f"{code_home_folder}logs/training/daily_data/MLP_log_{current_time()}"
    data_folder = "data/train_test_arrays/"
    filename = f"{code_home_folder}models/MLP_model_daily{arch}.sav"

### Code ###


write(title, f"{current_time()}\nWEATHER TRAINING DATA\n")
write(title, f"All buildings. Daily data, including meta data.")

write(title, "\nBUILDING TRAINING DATA\n")
write(title, f"All buildings, daily total energy.")

print("Importing data...")
X_train = torch.from_numpy(np.load(glob.glob(f"{code_home_folder}{data_folder}X_train.npy")[0]))
y_train = torch.from_numpy(np.load(glob.glob(f"{code_home_folder}{data_folder}y_train.npy")[0]))
X_test = torch.from_numpy(np.load(glob.glob(f"{code_home_folder}{data_folder}X_test.npy")[0]))
y_test = torch.from_numpy(np.load(glob.glob(f"{code_home_folder}{data_folder}y_test.npy")[0]))

write(title, f"Training array dimensions: {X_train.shape} {y_train.shape}")
write(title, f"Test array dimensions: {X_test.shape} {y_test.shape}")

print("Processing data...")

