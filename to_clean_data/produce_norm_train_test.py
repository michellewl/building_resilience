import glob
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


code_home_folder = "/home/mwlw3/Documents/Guided_team_challenge/building_resilience/"
data_folder = "data/processed_arrays/"
save_folder = "data/train_test_arrays/"


print("WEATHER TRAINING DATA")
print("Reading dataset...")
files = glob.glob(f"{code_home_folder}{data_folder}monthly_weather.csv")
weather_array = np.genfromtxt(files[0], delimiter=",")
print("Done.")



print("\nBUILDING TRAINING DATA")
print("Reading dataset...")
files = glob.glob(f"{code_home_folder}{data_folder}monthly_energy.csv")
energy_array = np.genfromtxt(files[0], delimiter=",")
print("Done.")

X = weather_array
y = energy_array

print("Splitting into train and test sets...")
test_size = 0.15
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print("Applying normalisation...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
y_train = scaler.fit_transform(y_train)
X_test = scaler.transform(X_test)
y_test = scaler.transform(y_test)


np.savetxt(f"{code_home_folder}{save_folder}X_train.csv", X_train, delimiter=",")
np.savetxt(f"{code_home_folder}{save_folder}y_train.csv", y_train, delimiter=",")
np.savetxt(f"{code_home_folder}{save_folder}X_test.csv", X_test, delimiter=",")
np.savetxt(f"{code_home_folder}{save_folder}y_test.csv", y_test, delimiter=",")

print("Saved train/test files.")

