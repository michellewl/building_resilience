import glob
import numpy as np
from sklearn.neural_network import MLPRegressor
from functions.functions import write, current_time
import pickle

windows_os = True

arch = "_50_50"

if windows_os:
    code_home_folder = "C:\\Users\\Michelle\\OneDrive - University of Cambridge\\MRes\\Guided_Team_Challenge\\building_resilience\\"
    title = f"{code_home_folder}logs\\training\\daily_data\\MLP_pytorch_log_{current_time()}"
    data_folder = "data\\train_test_arrays\\"
    filename = f"{code_home_folder}models\\MLP_model_daily{arch}.sav"
    title = f"{code_home_folder}logs\\training\\daily_data\\MLP{arch}_log_{current_time()}"
else:
    code_home_folder = "/home/mwlw3/Documents/Guided_team_challenge/building_resilience/"
    title = f"{code_home_folder}logs/training/daily_data/MLP_log_{current_time()}"
    data_folder = "data/train_test_arrays/"
    filename = f"{code_home_folder}models/MLP_model_daily{arch}.sav"
    title = f"{code_home_folder}logs/training/daily_data/MLP{arch}_log_{current_time()}"


write(title, f"{current_time()}\nWEATHER TRAINING DATA\n")
write(title, f"All buildings. Daily data, including meta data.")

write(title, "\nBUILDING TRAINING DATA\n")
write(title, f"All buildings, daily total energy.")

print("Importing data...")
X_train = np.load(f"{code_home_folder}{data_folder}X_train.npy")
y_train = np.load(f"{code_home_folder}{data_folder}y_train.npy")
X_test = np.load(f"{code_home_folder}{data_folder}X_test.csv")
y_test = np.load(f"{code_home_folder}{data_folder}y_test.csv")

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


pickle.dump(model, open(filename, "wb"))
print("Saved model.")
