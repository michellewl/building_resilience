## This script uses the PyTorch machine learning library to train a feed-forward neural network model
## and saves the model as a .tar file. This can be loaded using PyTorch to continue training or evaluate
## the model at a later stage.
## Note that the network still requires separate definition from the saved model.

## Inputs: 6 numpy files (train, validation and test for inputs and targets)
## Outputs: 1 model .tar file (contains optimised weights, loss history for training and validation sets, number of epochs)

### Package imports

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from building_dataset import BuildingDataset
from multilayer_perceptron import SimpleNet_3bn, SimpleNet_3
from copy import deepcopy

### Hyperparameters

#### Define network architecture

all_hidden_layers = 50

hidden_layer_1 = all_hidden_layers
hidden_layer_2 = all_hidden_layers
hidden_layer_3 = all_hidden_layers
#hidden_layer_4 = all_hidden_layers

#### File naming system for models of different architecture

arch = f"_{hidden_layer_1}_{hidden_layer_2}_{hidden_layer_3}"#_{hidden_layer_4}"

#### Define batch size, maximum number of epochs, and number of batches per print statement (during training).

batch_size = 16
num_epochs = 1000
batches_per_print = 5000

### (Future work) Setting a random seed for reproducibility

torch.manual_seed(7)

### Folder formatting for different operating systems

windows_os = True

if windows_os:
    code_home_folder = "C:\\Users\\Michelle\\OneDrive - University of Cambridge\\MRes\\Guided_Team_Challenge\\building_resilience\\"
    data_folder = "data\\ashrae-energy-prediction\\train_test_arrays\\"
    filename = f"{code_home_folder}models\\neural_networks\\saved\\MLP_pytorch_model_daily{arch}_take7_no_bn.tar"
else:
    code_home_folder = "/home/mwlw3/Documents/Guided_team_challenge/building_resilience/"
    data_folder = "data/ashrae-energy-prediction/train_test_arrays/"
    filename = f"{code_home_folder}models/neural_networks/saved/MLP_pytorch_model_daily{arch}.tar"

### Code


### Load data

X_train_filepath = f"{code_home_folder}{data_folder}X_train.npy"
y_train_filepath = f"{code_home_folder}{data_folder}y_train.npy"
X_validation_filepath = f"{code_home_folder}{data_folder}X_val.npy"
y_validation_filepath = f"{code_home_folder}{data_folder}y_val.npy"


#### PyTorch uses dataset classes, the BuildingDataset used here is defined in a separate script.

training_dataset = BuildingDataset(X_train_filepath, y_train_filepath)
validation_dataset = BuildingDataset(X_validation_filepath, y_validation_filepath)

print(f"Training dataset size:{len(training_dataset)}"
      f"\nValidation dataset size:{len(validation_dataset)}")

#### PyTorch uses a DataLoader to load data in batches from a Dataset during training.

training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

### Define neural network model

#### The SimpleNet network is a PyTorch class, defined in a separate script.

simple_net = SimpleNet_3(int(training_dataset.nfeatures()), hidden_layer_1, hidden_layer_2, hidden_layer_3)

print(simple_net)

#### Define the loss criterion and neural network optimiser.

criterion = nn.MSELoss()
optimiser = optim.Adam(simple_net.parameters())#, weight_decay=)

### Train network

#### We will store loss histories to plot in another script.

training_loss_history = []
val_loss_history = []

print("Begin training...")
for epoch in range(num_epochs):
    running_loss = 0.0 # for printing
    loss_sum = 0 # for storing

    # Training batches
    simple_net.train()
    for batch_num, data in enumerate(training_dataloader):
        inputs_training = data["inputs"]
        targets_training = data["targets"]

        optimiser.zero_grad()

        outputs_training = simple_net(inputs_training)
        loss = criterion(outputs_training, targets_training)
        loss.backward()
        optimiser.step()

        # print statistics
        running_loss += loss.item()
        if batch_num % batches_per_print == 0:  # print every x mini-batches as defined at the top of the script
            print(f"Epoch {epoch} batch {batch_num} loss: {running_loss / batches_per_print}")
            running_loss = 0.0

        loss_sum += loss.item()*batch_size
    training_loss_history.append(loss_sum/len(training_dataset))

    # Validation batches
    simple_net.eval()
    validation_loss_sum = 0
    with torch.no_grad():
        for batch_num, data in enumerate(validation_dataloader):
            inputs_val = data["inputs"]
            targets_val = data["targets"]

            outputs_val = simple_net(inputs_val)
            loss = criterion(outputs_val, targets_val)
            validation_loss_sum += loss.item() * batch_size
    val_loss_this_epoch = validation_loss_sum / len(validation_dataset)

	#### Store the model parameters giving the lowest validation loss.
    if  epoch == 0 or val_loss_this_epoch < min(val_loss_history):
        best_weights = deepcopy(simple_net.state_dict())
        best_epoch = epoch # for pointing out the on plot later
    print(f"Epoch {epoch} validation loss: {val_loss_this_epoch}")
    val_loss_history.append(val_loss_this_epoch)

	#### Save the model every 2 epochs of training.
	#### We can access the latest saved model while the network is training or interrupt the training as desired.
    if epoch % 2 == 0:
        torch.save({
            'total_epochs': epoch,
            'final_state_dict': simple_net.state_dict(),
            'optimiser_state_dict': optimiser.state_dict(),
            'training_loss_history': training_loss_history,
            "best_state_dict": best_weights,
            "best_epoch": best_epoch,
            "validation_loss_history": val_loss_history
        },
            filename)

print('Finished Training')
