import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import glob
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from functions import show_data
from functions import nan_mean_interpolation, nan_count_total, nan_count_by_variable, write, \
    get_building_ids, fix_time_gaps, wind_direction_trigonometry, current_time
import datetime as dt

#folder = "/space/mwlw3/GTC_data_exploration/"
folder = "C:\\Users\\Michelle\\PycharmProjects\\GTC\\"

now = current_time()
title = f"lin_reg_log_{now}"

print("WEATHER TRAINING DATA")
write(title, "WEATHER TRAINING DATA\n")
print("Reading dataset...")
files = glob.glob(f"{folder}weather_processed*buildings.csv")
weather_array = np.genfromtxt(files[0], delimiter=",")
write(title, "All buildings.")
write(title, f"Data shape: {weather_array.shape}")

weather_variables = ["air_temperature","dew_temperature","wind_speed", "cos_wind_direction", "sin_wind_direction"]

print("Done.")



print("\nBUILDING TRAINING DATA")
write(title, "\nBUILDING TRAINING DATA\n")
print("Reading dataset...")
files = glob.glob(f"{folder}energy_processed*buildings.csv")
energy_array = np.genfromtxt(files[0], delimiter=",")
write(title, "All buildings.")
write(title, f"Data shape: {energy_array.shape}")
print("Done.")

print("\nMODEL TRAIN")
write(title, "\nMODEL TRAIN\n")
X = weather_array
y = energy_array

print("Splitting into train and test sets...")
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
write(title, f"Training size: {1-test_size}")
write(title, f"Test size: {test_size}")

write(title, f"Training array dimensions: {X_train.shape} {y_train.shape}")
write(title, f"Test array dimensions: {X_test.shape} {y_test.shape}")

# write("\nBefore normalisation:")
# write(f"X_train mean, std: {X_train.mean()}, {X_train.std()}")
# write(f"y_train mean, std: {y_train.mean()}, {y_train.std()}")
# write(f"X_test mean, std: {X_test.mean()}, {X_test.std()}")
# write(f"y_test mean, std: {y_test.mean()}, {y_test.std()}")

print("Applying normalisation...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
y_train = scaler.fit_transform(y_train)
X_test = scaler.transform(X_test)
y_test = scaler.transform(y_test)

# write("\nAfter normalisation:")
# write(f"X_train mean, std: {X_train.mean()}, {X_train.std()}")
# write(f"y_train mean, std: {y_train.mean()}, {y_train.std()}")
# write(f"X_test mean, std: {X_test.mean()}, {X_test.std()}")
# write(f"y_test mean, std: {y_test.mean()}, {y_test.std()}")

write(title, "\nNormalised the training data and applied the same to the test set.")

print("Fitting linear regression model...")
model = LinearRegression(fit_intercept = False).fit(X_train,y_train)
print(f"R\u00b2: {model.score(X_train,y_train)}")
write(title, "\nLinear regression model fit:")
write(title, f"R\u00b2: {model.score(X_train,y_train)}")

y_train_predicted = model.predict(X_train)
print(f"Mean squared error: {mean_squared_error(y_train, y_train_predicted)}")
write(title, f"Mean squared error: {mean_squared_error(y_train, y_train_predicted)}")

print("\nMODEL TEST")
write(title, "\nLinear regression model test:")
print("Predicting the building results...")

y_predicted = model.predict(X_test)

# print(f"Dimensions of predicted building meter readings: {y_predicted.shape}")
# print(f"Dimensions of observed building meter readings: {y_test.shape}")

print(f"Mean squared error: {mean_squared_error(y_test, y_predicted)}")
write(title, f"Mean squared error: {mean_squared_error(y_test, y_predicted)}")
print("Model coefficients:")
write(title, "\nModel coefficients:")
coefficients = model.coef_.reshape(-1,1).tolist()
for i in range(len(weather_variables)):
    print(f"{weather_variables[i]} : {coefficients[i]}")
    write(title, f"{weather_variables[i]} : {coefficients[i]}")

print("Model intercept:")
print(model.intercept_)
write(title, "\nModel intercept:")
write(title, model.intercept_)