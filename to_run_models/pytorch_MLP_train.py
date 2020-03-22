from functions.functions import write, current_time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.building_dataset import BuildingDataset
from neural_networks.multilayer_perceptron import SimpleNet
from copy import deepcopy

### Folder formatting ###

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

### Code ###


# write(title, f"{current_time()}\nWEATHER TRAINING DATA\n")
# write(title, f"All buildings. Daily data, including meta data.")
#
# write(title, "\nBUILDING TRAINING DATA\n")
# write(title, f"All buildings, daily total energy.")


batch_size = 16
num_epochs = 1000
batches_per_print = 5000

### Run with the smaller test set for now to develop code ###
X_train_filepath = f"{code_home_folder}{data_folder}X_train.npy"
y_train_filepath = f"{code_home_folder}{data_folder}y_train.npy"
X_validation_filepath = f"{code_home_folder}{data_folder}X_val.npy"
y_validation_filepath = f"{code_home_folder}{data_folder}y_val.npy"

training_dataset = BuildingDataset(X_train_filepath, y_train_filepath)
validation_dataset = BuildingDataset(X_validation_filepath, y_validation_filepath)

print(f"Training dataset size:{len(training_dataset)}"
      f"\nValidation dataset size:{len(validation_dataset)}")

training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)



simple_net = SimpleNet(number_of_features=int(training_dataset.nfeatures()))

print(simple_net)

criterion = nn.MSELoss()
optimiser = optim.Adam(simple_net.parameters())#, weight_decay=)

epoch_losses = []
epoch_losses_validation = []

print("Begin training...")
for epoch in range(num_epochs):
    running_loss = 0.0
    loss_sum = 0
    # Training batches
    simple_net.train()
    for batch_num, data in enumerate(training_dataloader):
        inputs = data["inputs"]
        targets = data["targets"]


        optimiser.zero_grad()

        outputs = simple_net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimiser.step()

        # print statistics
        running_loss += loss.item()
        if batch_num % batches_per_print == 0:  # print every 2000 mini-batches
            print(f"Epoch {epoch} batch {batch_num} loss: {running_loss / batches_per_print}")
            running_loss = 0.0

        loss_sum += loss.item()*batch_size
    epoch_losses.append(loss_sum/len(training_dataset))

    # Validation batches
    simple_net.eval()
    validation_loss_sum = 0
    with torch.no_grad():
        for batch_num, data in enumerate(validation_dataloader):
            validation_inputs = data["inputs"]
            validation_targets = data["targets"]


            validation_outputs = simple_net(validation_inputs)
            validation_loss = criterion(validation_outputs, validation_targets)
            validation_loss_sum += validation_loss.item() * batch_size
    epoch_validation_loss = validation_loss_sum / len(validation_dataset)

    if  epoch == 0 or epoch_validation_loss < min(epoch_losses_validation):
        best_weights = deepcopy(simple_net.state_dict())
        best_epoch = epoch
    print(f"Epoch {epoch} validation loss: {epoch_validation_loss}")
    epoch_losses_validation.append(epoch_validation_loss)

    if epoch % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': simple_net.state_dict(),
            'optimiser_state_dict': optimiser.state_dict(),
            #'loss': loss,
            'loss_history': epoch_losses,
            "best_weights": best_weights,
            "best_epoch": best_epoch,
            "validation_loss_history": epoch_losses_validation
        },
            filename)

print('Finished Training')