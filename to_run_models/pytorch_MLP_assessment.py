import numpy as np
from sklearn.metrics import mean_squared_error
from functions.functions import write, current_time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.building_dataset import BuildingDataset
from neural_networks.multilayer_perceptron import SimpleNet
from copy import deepcopy

windows_os = True

arch = "_100_100"

if windows_os:
    code_home_folder = "C:\\Users\\Michelle\\OneDrive - University of Cambridge\\MRes\\Guided_Team_Challenge\\building_resilience\\"
    title = f"{code_home_folder}logs\\assessment\\daily_data\\MLP_pytorch_log_{current_time()}"
    data_folder = "data\\train_test_arrays\\"
    filename = f"{code_home_folder}models\\MLP_pytorch_model_daily{arch}.tar"
else:
    code_home_folder = "/home/mwlw3/Documents/Guided_team_challenge/building_resilience/"
    title = f"{code_home_folder}logs/assessment/daily_data/MLP_pytorch_log_{current_time()}"
    data_folder = "data/train_test_arrays/"
    filename = f"{code_home_folder}models/MLP_pytorch_model_daily{arch}.tar"

batch_size = 16



print("Importing data...")
#X_train = np.load(f"{code_home_folder}{data_folder}X_train.npy")
#y_train = np.load(f"{code_home_folder}{data_folder}y_train.npy")
X_test_filepath = f"{code_home_folder}{data_folder}X_test.npy"
y_test_filepath = f"{code_home_folder}{data_folder}y_test.npy"

test_dataset = BuildingDataset(X_test_filepath, y_test_filepath)
print(f"Test dataset size: {len(test_dataset)}")
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


print("Importing model...")
checkpoint = torch.load(filename)
best_weights = checkpoint["best_weights"]

simple_net = SimpleNet(number_of_features=int(test_dataset.nfeatures()))
simple_net.load_state_dict(best_weights)
print(simple_net)
simple_net.eval()


# Model prediction

criterion = nn.MSELoss()
loss_sum = 0
for batch_num, data in enumerate(test_dataloader):
    inputs = data["inputs"]
    targets = data["targets"]

    outputs = simple_net(inputs)
    loss = criterion(outputs, targets)

    # # print statistics
    # running_loss += loss.item()
    # if batch_num % batches_per_print == 0:  # print every 2000 mini-batches
    #     print(f"Epoch {epoch} batch {batch_num} loss: {running_loss / batches_per_print}")
    #     running_loss = 0.0
    loss_sum += loss.item() * batch_size

final_loss = loss_sum / len(test_dataset)
print(f"Test set MSE: {final_loss}")

best_loss = min(checkpoint["loss_history"])
epochs = checkpoint["epoch"]
best_val_loss = min(checkpoint["validation_loss_history"])

write(title, f"MLP model uses weather variables and building meta data.\n")
write(title, f"\nArchitecture: {arch}"
             f"\nBest training loss: {best_loss}"
             f"\nBest validation loss: {best_val_loss}"
             f"\nIterations: {epochs}"
             f"\nActivation function: relu"
             f"\nSolver: adam")

write(title, f"Test set MSE: {final_loss}")

