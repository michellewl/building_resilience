import glob
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from functions.functions import write, current_time
import pickle

code_home_folder = "/home/mwlw3/Documents/Guided_team_challenge/building_resilience/"

title = f"{code_home_folder}logs/training/daily_data/MLP_log_{current_time()}"

data_folder = "data/train_test_arrays/"





write(title, f"{current_time()}\nWEATHER TRAINING DATA\n")
write(title, f"All buildings. Daily data, including meta data.")

write(title, "\nBUILDING TRAINING DATA\n")
write(title, f"All buildings, daily total energy.")

print("Importing data...")
X_train = np.genfromtxt(glob.glob(f"{code_home_folder}{data_folder}X_train.csv")[0], delimiter=",")
y_train = np.genfromtxt(glob.glob(f"{code_home_folder}{data_folder}y_train.csv")[0], delimiter=",")
X_test = np.genfromtxt(glob.glob(f"{code_home_folder}{data_folder}X_test.csv")[0], delimiter=",")
y_test = np.genfromtxt(glob.glob(f"{code_home_folder}{data_folder}y_test.csv")[0], delimiter=",")

write(title, f"Training array dimensions: {X_train.shape} {y_train.shape}")
write(title, f"Test array dimensions: {X_test.shape} {y_test.shape}")

print("Fitting multi-layer perceptron regression model...")
model = MLPRegressor(hidden_layer_sizes=(50,50,), batch_size=16, verbose=True, max_iter=200)
write(title, f"{current_time()}\nArchitecture: {model.hidden_layer_sizes}\nBatch size: {model.batch_size}")
model.fit(X_train,y_train)

write(title, f"\n{current_time()}\n\nBest loss: {model.best_loss_}"
             f"\nIterations: {model.n_iter_}"
             f"\nLayers: {model.n_layers_}"
             f"\nOutputs: {model.n_outputs_}"
             f"\nActivation function: {model.activation}"
             f"\nOutput activation function: {model.out_activation_}"
             f"\nSolver: {model.solver}")

arch = "_50_50"
filename = f"{code_home_folder}models/MLP_model_daily{arch}.sav"
pickle.dump(model, open(filename, "wb"))
print("Saved model.")
