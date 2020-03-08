import glob
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from functions import write, current_time
import pickle

folder = "/space/mwlw3/GTC_data_exploration/data_train_test/"
#folder = "C:\\Users\\Michelle\\PycharmProjects\\GTC\\data_train_test\\"

now = current_time()
title = f"MLP_log_{now}"

data_type = "buildings"

# print("WEATHER TRAINING DATA")
write(title, f"{current_time()}\nWEATHER TRAINING DATA\n")
# print("Reading dataset...")
# files = glob.glob(f"{folder}weather_only_processed*{data_type}.csv")
# weather_array = np.genfromtxt(files[0], delimiter=",")
write(title, f"All {data_type}. Does not include meta data")

vars = "_no_meta"

# write(title, f"Data shape: {weather_array.shape}")
# weather_array = weather_array[:,:5]
# print(f"New weather_array shape (should be 5 columns): {weather_array.shape}")

# #weather_variables = ["air_temperature","dew_temperature","wind_speed", "cos_wind_direction", "sin_wind_direction"]

# print("Done.")



#print("\nBUILDING TRAINING DATA")
write(title, "\nBUILDING TRAINING DATA\n")
# print("Reading dataset...")
# files = glob.glob(f"{folder}energy_processed*{data_type}.csv")
# energy_array = np.genfromtxt(files[0], delimiter=",")
write(title, f"All {data_type}.")
# write(title, f"Data shape: {energy_array.shape}")
# print("Done.")

# print("\nMODEL TRAIN")
# write(title, "\nMODEL TRAIN\n")
# X = weather_array
# y = energy_array

# print("Splitting into train and test sets...")
# test_size = 0.33
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
# write(title, f"Training size: {1-test_size}")
# write(title, f"Test size: {test_size}")



print("Importing data...")
X_train = np.genfromtxt(glob.glob(f"{folder}X_train{vars}.csv")[0], delimiter=",")
y_train = np.genfromtxt(glob.glob(f"{folder}y_train{vars}.csv")[0], delimiter=",")
X_test = np.genfromtxt(glob.glob(f"{folder}X_test{vars}.csv")[0], delimiter=",")
y_test = np.genfromtxt(glob.glob(f"{folder}y_test{vars}.csv")[0], delimiter=",")

write(title, f"Training array dimensions: {X_train.shape} {y_train.shape}")
write(title, f"Test array dimensions: {X_test.shape} {y_test.shape}")

#write(title, f"\nNormalised the training data and applied the same to the test set.")

print("Fitting multi-layer perceptron regression model...")
model = MLPRegressor(hidden_layer_sizes=(100,100,100,), batch_size=16, verbose=True, max_iter=4)
write(title, f"{current_time()}\nArchitecture: {model.hidden_layer_sizes}\nBatch size: {model.batch_size}")
model.fit(X_train,y_train)

write(title, f"\n{current_time()}\n\nBest loss: {model.best_loss_}"
             f"\nIterations: {model.n_iter_}"
             f"\nLayers: {model.n_layers_}"
             f"\nOutputs: {model.n_outputs_}"
             f"\nActivation function: {model.activation}"
             f"\nOutput activation function: {model.out_activation_}"
             f"\nSolver: {model.solver}")
#pd.DataFrame(model.loss_curve_).plot()

arch = "_100_100_100"
filename = f"/space/mwlw3/GTC_data_exploration/to_run_models/MLP_model{vars}{arch}.sav"
pickle.dump(model, open(filename, "wb"))
# np.savetxt(f"X_train{vars}.csv", X_train, delimiter=",")
# np.savetxt(f"y_train{vars}.csv", y_train, delimiter=",")
# np.savetxt(f"X_test{vars}.csv", X_test, delimiter=",")
# np.savetxt(f"y_test{vars}.csv", y_test, delimiter=",")

print("Saved model.")

# print(f"R\u00b2: {model.score(X_train,y_train)}")
# write_nn("\nMLP regression model fit:")
# write_nn(f"R\u00b2: {model.score(X_train,y_train)}")
#


# y_train_predicted = model.predict(X_train[0:])


# print(f"Mean squared error: {mean_squared_error(y_train, y_train_predicted)}")
# write(title, f"Mean squared error: {mean_squared_error(y_train, y_train_predicted)}")

# print("\nMODEL TEST")
# write(title, "\nMLP regression model test:")
# print("Predicting the building results...")

# y_predicted = model.predict(X_test)

# # print(f"Dimensions of predicted building meter readings: {y_predicted.shape}")
# # print(f"Dimensions of observed building meter readings: {y_test.shape}")

# print(f"Mean squared error: {mean_squared_error(y_test, y_predicted)}")
# write(title, f"Mean squared error: {mean_squared_error(y_test, y_predicted)}")

# # write_nn(f"\nMLP parameters: {model.get_params}")
