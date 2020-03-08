import glob
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from functions import write, current_time
import pickle

folder = "/space/mwlw3/GTC_data_exploration/"
#folder = "C:\\Users\\Michelle\\PycharmProjects\\GTC\\"

vars = "_yrbt_sqft"
arch = "_50_50"
now = current_time()
title = f"MLP_assessment_log{vars}{arch}_{now}"

print("Importing data...")
X_train = np.genfromtxt(glob.glob(f"{folder}X_train{vars}.csv")[0], delimiter=",")
y_train = np.genfromtxt(glob.glob(f"{folder}y_train{vars}.csv")[0], delimiter=",")
X_test = np.genfromtxt(glob.glob(f"{folder}X_test{vars}.csv")[0], delimiter=",")
y_test = np.genfromtxt(glob.glob(f"{folder}y_test{vars}.csv")[0], delimiter=",")

print("Importing model...")
filename = f"MLP_model{vars}{arch}.sav"
model = pickle.load(open(filename, 'rb'))

write(title, f"MLP model uses weather variables and building meta data (year built and square feet).\n")
write(title, f"\nArchitecture: {model.hidden_layer_sizes}\nBatch size: {model.batch_size}")
write(title, f"\nBest loss: {model.best_loss_}"
             f"\nIterations: {model.n_iter_}"
             f"\nLayers: {model.n_layers_}"
             f"\nOutputs: {model.n_outputs_}"
             f"\nActivation function: {model.activation}"
             f"\nOutput activation function: {model.out_activation_}"
             f"\nSolver: {model.solver}")
#pd.DataFrame(model.loss_curve_).plot()

write(title, f"Training array dimensions: {X_train.shape} {y_train.shape}")
write(title, f"Test array dimensions: {X_test.shape} {y_test.shape}")

## TRAINING SET
print("Training set:")
dataset_X = X_train
dataset_y = y_train
n_batches = 100
batch_size = int(dataset_X.shape[0] / n_batches) 
array_list = []
for i in range(0, n_batches):
    print(f"Training batch #{i+1}")
    if (i==n_batches):
        y_predicted = model.predict(dataset_X[i*batch_size:])
        mse = mean_squared_error(dataset_y[i*batch_size:], y_predicted)
        total_se = mse*y_predicted.shape[0]
    else: 
        y_predicted = model.predict(dataset_X[i*batch_size:(i+1)*batch_size])
        mse = mean_squared_error(dataset_y[i*batch_size:(i+1)*batch_size], y_predicted)
        total_se = mse*y_predicted.shape[0]
    array_list.append(total_se)
all_se = np.concatenate(array_list, axis=None)
all_se = all_se.sum()
mean_se = all_se / dataset_X.shape[0]

print(f"Mean squared error (training data): {mean_se}")
write(title, f"\nMean squared error (training data): {mean_se}")


## TEST SET
print("Test set:")
dataset_X = X_test
dataset_y = y_test
n_batches = 100
batch_size = int(dataset_X.shape[0] / n_batches) 
array_list = []
for i in range(0, n_batches):
    print(f"Test batch #{i+1}")
    if (i==n_batches):
        y_predicted = model.predict(dataset_X[i*batch_size:])
        mse = mean_squared_error(dataset_y[i*batch_size:], y_predicted)
        total_se = mse*y_predicted.shape[0]
    else: 
        y_predicted = model.predict(dataset_X[i*batch_size:(i+1)*batch_size])
        mse = mean_squared_error(dataset_y[i*batch_size:(i+1)*batch_size], y_predicted)
        total_se = mse*y_predicted.shape[0]
    array_list.append(total_se)
all_se = np.concatenate(array_list, axis=None)
all_se = all_se.sum()
mean_se = all_se / dataset_X.shape[0]

print(f"Mean squared error (test data): {mean_se}")
write(title, f"Mean squared error (test data): {mean_se}")



