# This script takes a pre-filled evaluation table (manually compiled during project progress) and plots the SMAPE
# evaluation metric. It also calculates the number of parameters for the neural network models (varies with architecture)
# to represent model complexity.

# Inputs: Model evaluation table
# This is compiled  manually.

# Outputs: Evaluation scatter plot (saved as .png file)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from classes import NNetwork


windows_os = True

if windows_os:
    code_home_folder = "C:\\Users\\Michelle\\OneDrive - University of Cambridge\\MRes\\Guided_Team_Challenge\\building_resilience\\"
    folder = f"{code_home_folder}logs\\assessment\\daily_data\\"
    models_folder = f"{code_home_folder}models\\neural_networks\\saved\\"
else:
    code_home_folder = "/home/mwlw3/Documents/Guided_team_challenge/building_resilience/"
    folder = f"{code_home_folder}logs/assessment/daily_data/"
    models_folder = f"{code_home_folder}models/neural_networks/saved/"

filepath = f"{folder}Model_evaluation.csv"

eval_table = pd.read_csv(filepath)

models_list = glob(f"{models_folder}*.tar")

# Create new column in evaluation table dataframe
eval_table["n_parameters"] = ""

# Fill the table for the linear regression model
eval_table.loc[eval_table.model_type == "linear_regression", "n_parameters"] = 49+1

# Fill the table for the MLP models
for model_name in models_list:
    model = NNetwork(model_name)
    eval_table.loc[(eval_table.model_type == "multilayer_perceptron")
                   & (eval_table.hidden_layers == model.hidden_layers)
                   & (eval_table.nodes == model.nodes[0]), "n_parameters"] = model.n_parameters

# Use a log scale
eval_table["log_n_params"] = np.log(eval_table.n_parameters.astype('float64'))

# Create plot

fig, ax = plt.subplots()
ax.scatter(eval_table.loc[eval_table.model_type == "multilayer_perceptron"].log_n_params,
           eval_table.loc[eval_table.model_type == "multilayer_perceptron"].smape, label="neural network")
ax.scatter(eval_table.loc[eval_table.model_type == "linear_regression"].log_n_params,
           eval_table.loc[eval_table.model_type == "linear_regression"].smape, label="linear regression")
plt.ylim(0, 100)
plt.ylabel("symmetric mean absolute percentage error")
plt.xlabel("log(number of parameters)")
plt.title("% error")
plt.legend()

# Save plot figure
fig.savefig(f"{folder}model_evaluation.png", dpi=200)