import numpy as np
from sklearn.metrics import mean_squared_error
from functions.functions import write, current_time
from functions.metrics import cv_metric, rmse, smape
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.building_dataset import BuildingDataset
from neural_networks.multilayer_perceptron import SimpleNet, SimpleNet_3, SimpleNet_4
from copy import deepcopy

windows_os = True

all_hidden_layers = 25

hidden_layer_1 = all_hidden_layers
hidden_layer_2 = all_hidden_layers
hidden_layer_3 = all_hidden_layers
hidden_layer_4 = all_hidden_layers

arch = f"_{hidden_layer_1}_{hidden_layer_2}_{hidden_layer_3}_{hidden_layer_4}"

if windows_os:
    code_home_folder = "C:\\Users\\Michelle\\OneDrive - University of Cambridge\\MRes\\Guided_Team_Challenge\\building_resilience\\"
    title = f"{code_home_folder}logs\\assessment\\daily_data\\MLP_pytorch_log_{current_time()}"
    data_folder = "data\\train_test_arrays\\"
    filename = f"{code_home_folder}models\\saved\\MLP_pytorch_model_daily{arch}.tar"
else:
    code_home_folder = "/home/mwlw3/Documents/Guided_team_challenge/building_resilience/"
    title = f"{code_home_folder}logs/assessment/daily_data/MLP_pytorch_log_{current_time()}"
    data_folder = "data/train_test_arrays/"
    filename = f"{code_home_folder}models/saved/MLP_pytorch_model_daily{arch}.tar"

batch_size = 16



print("Importing data...")
X_test_filepath = f"{code_home_folder}{data_folder}X_test.npy"
y_test_filepath = f"{code_home_folder}{data_folder}y_test.npy"

test_dataset = BuildingDataset(X_test_filepath, y_test_filepath)
print(f"Test dataset size: {len(test_dataset)}")
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


print("Importing model...")
checkpoint = torch.load(filename)

simple_net = SimpleNet(int(test_dataset.nfeatures()), hidden_layer_1, hidden_layer_2, hidden_layer_3, hidden_layer_4)
simple_net.load_state_dict(checkpoint["best_state_dict"], strict=True)
print(simple_net)
simple_net.eval()


# Model prediction

target_list = []
prediction_list = []

for batch_num, data in enumerate(test_dataloader):
    inputs = data["inputs"]
    targets = data["targets"]

    outputs = simple_net(inputs)

    target_list.append(targets.detach().numpy())
    prediction_list.append(outputs.detach().numpy())

targets_array = np.concatenate(target_list, axis=None)
prediction_array = np.concatenate(prediction_list, axis=None)

### Get metrics ###

cv_test_set = cv_metric(targets_array, prediction_array)
rmse_test_set = rmse(targets_array, prediction_array)
smape_test_set = smape(targets_array, prediction_array)


print(f"Test set RMSE: {rmse_test_set}\nTest set coefficient of variation: {cv_test_set}"
      f"\nTest set SMAPE: {smape_test_set}")

best_loss = min(checkpoint["training_loss_history"])
epochs = checkpoint["total_epochs"]
best_val_loss = min(checkpoint["validation_loss_history"])

write(title, f"MLP model uses weather variables and building meta data.\n")
write(title, f"\nArchitecture: {hidden_layer_1}, {hidden_layer_2}, {hidden_layer_3}, {hidden_layer_4}"
             f"\nEpochs: {epochs}"
             f"\nActivation function: relu"
             f"\nLoss function: Mean Squared Error"
             f"\nSolver: adam"
             f"\nBest training loss: {best_loss}"
             f"\nBest validation loss: {best_val_loss}"
             f"\nTest set MSE: {mean_squared_error(targets_array, prediction_array)}")

write(title, f"\nTest set RMSE: {rmse_test_set}\nTest set coefficient of variation: {cv_test_set}"
             f"\nTest set SMAPE: {smape_test_set}"
             f"\nTest set MAE: {np.mean(np.abs(targets_array - prediction_array))}")

write(title, f"\n\n\n {simple_net}")

