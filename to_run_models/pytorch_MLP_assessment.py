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
    title = f"{code_home_folder}logs\\training\\daily_data\\MLP_pytorch_log_{current_time()}"
    data_folder = "data\\train_test_arrays\\"
    filename = f"{code_home_folder}models\\MLP_pytorch_model_daily{arch}.tar"
else:
    code_home_folder = "/home/mwlw3/Documents/Guided_team_challenge/building_resilience/"
    title = f"{code_home_folder}logs/training/daily_data/MLP_log_{current_time()}"
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
best_weights = torch.load(filename)["best_weights"]

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


#
# write(title, f"MLP model uses weather variables and building meta data.\n")
# write(title, f"\nArchitecture: {arch}"
#              f"\nBest loss: {model.best_loss_}"
#              f"\nIterations: {model.n_iter_}"
#              f"\nLayers: {model.n_layers_}"
#              f"\nOutputs: {model.n_outputs_}"
#              f"\nActivation function: {model.activation}"
#              f"\nOutput activation function: {model.out_activation_}"
#              f"\nSolver: {model.solver}")
#
# write(title, f"Training array dimensions: {X_train.shape} {y_train.shape}")
# write(title, f"Test array dimensions: {X_test.shape} {y_test.shape}")
#
# ## TRAINING SET
# print("Training set:")
# dataset_X = X_train
# dataset_y = y_train
# n_batches = 10
# batch_size = int(dataset_X.shape[0] / n_batches)
# array_list = []
# for i in range(0, n_batches):
#     print(f"Training batch #{i+1}")
#     if (i==n_batches):
#         y_predicted = model.predict(dataset_X[i*batch_size:])
#         mse = mean_squared_error(dataset_y[i*batch_size:], y_predicted)
#         total_se = mse*y_predicted.shape[0]
#     else:
#         y_predicted = model.predict(dataset_X[i*batch_size:(i+1)*batch_size])
#         mse = mean_squared_error(dataset_y[i*batch_size:(i+1)*batch_size], y_predicted)
#         total_se = mse*y_predicted.shape[0]
#     array_list.append(total_se)
# all_se = np.concatenate(array_list, axis=None)
# all_se = all_se.sum()
# mean_se = all_se / dataset_X.shape[0]
#
# print(f"Mean squared error (training data): {mean_se}")
# write(title, f"\nMean squared error (training data): {mean_se}")
#
#
# ## TEST SET
# print("Test set:")
# dataset_X = X_test
# dataset_y = y_test
# n_batches = 10
# batch_size = int(dataset_X.shape[0] / n_batches)
# array_list = []
# for i in range(0, n_batches):
#     print(f"Test batch #{i+1}")
#     if (i==n_batches):
#         y_predicted = model.predict(dataset_X[i*batch_size:])
#         mse = mean_squared_error(dataset_y[i*batch_size:], y_predicted)
#         total_se = mse*y_predicted.shape[0]
#     else:
#         y_predicted = model.predict(dataset_X[i*batch_size:(i+1)*batch_size])
#         mse = mean_squared_error(dataset_y[i*batch_size:(i+1)*batch_size], y_predicted)
#         total_se = mse*y_predicted.shape[0]
#     array_list.append(total_se)
# all_se = np.concatenate(array_list, axis=None)
# all_se = all_se.sum()
# mean_se = all_se / dataset_X.shape[0]
#
# print(f"Mean squared error (test data): {mean_se}")
# write(title, f"Mean squared error (test data): {mean_se}")
#
#
#
