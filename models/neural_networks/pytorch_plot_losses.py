## This script uses the Matplotlib library to plot the loss history of a feed-forward neural network model (PyTorch).
## It takes a saved .tar file that is generated during network training, and can be run during or after training
## since it doesn't interfere with the training script. This is useful for determining whether it is appropriate to
## stop network training early.

## Inputs: 1 model .tar file (contains optimised weights, loss history for training and validation sets, number of epochs)
## This is the output from the training script (also in this folder).

## Outputs: 1 line plot figure (not saved automatically)
## This can be saved using the plot viewer.

### Package imports

import torch
import matplotlib.pyplot as plt
import numpy as np

### Hyperparameters

#### Define network architecture

all_hidden_layers = 50

hidden_layer_1 = all_hidden_layers
hidden_layer_2 = all_hidden_layers
hidden_layer_3 = all_hidden_layers
#hidden_layer_4 = all_hidden_layers

#### File naming system for models of different architecture
arch = f"_{hidden_layer_1}_{hidden_layer_2}_{hidden_layer_3}"#_{hidden_layer_4}"

windows_os = True

if windows_os:
    code_home_folder = "C:\\Users\\Michelle\\OneDrive - University of Cambridge\\MRes\\Guided_Team_Challenge\\building_resilience\\"
    filename = f"{code_home_folder}models\\neural_networks\\saved\\MLP_pytorch_model_daily{arch}_take3_no_bn.tar"
else:
    code_home_folder = "/home/mwlw3/Documents/Guided_team_challenge/building_resilience/"
    filename = f"{code_home_folder}models/neural_networks/saved/MLP_pytorch_model_daily{arch}.tar"

# Load loss history from saved model
training_losses = torch.load(filename)["training_loss_history"]
val_losses = torch.load(filename)["validation_loss_history"]
best_epoch = torch.load(filename)["best_epoch"]

# Plot training and validation loss history and annotate the epoch with best validation loss

fig, ax = plt.subplots()
ax.plot(range(len(training_losses)), np.log(training_losses), label="training loss")
ax.plot(range(len(val_losses)), np.log(val_losses), label="validation loss")
ax.scatter(best_epoch, min(np.log(val_losses)))
plt.annotate(f"epoch {best_epoch}", (best_epoch*1.05, min(np.log(val_losses))))
#plt.ylim(0, 1e-8)
plt.legend()
plt.xlabel("epoch")
plt.ylabel("log(loss)")
plt.title(f"Hidden layer nodes: {hidden_layer_1}, {hidden_layer_2}, {hidden_layer_3}")#, {hidden_layer_4}")
plt.show()