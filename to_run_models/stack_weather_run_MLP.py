import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from functions.functions import nan_mean_interpolation, nan_count_total, nan_count_by_variable, write, \
    get_building_ids, fix_time_gaps, wind_direction_trigonometry
import datetime as dt

from sklearn.neural_network import MLPRegressor

from functions.functions import write, current_time
import pickle

folder = "/space/mwlw3/GTC_data_exploration/data_ashrae_raw/"
#folder = "C:\\Users\\Michelle\\PycharmProjects\\GTC\\data\\data_ashrae_raw\\"

print("\nBUILDING META DATA\n")
files = glob.glob(f"/space/mwlw3/GTC_data_exploration/data_ashrae_raw/*meta*.csv")
meta_data = pd.read_csv(files[0])

start = dt.datetime(day=1, month=1, year=2016, hour=0, minute=0)
end = dt.datetime(day=31, month=12, year=2016, hour=23, minute=0)


print("WEATHER TRAINING DATA")
#write("WEATHER TRAINING DATA\n")
print("Reading dataset...")
files = glob.glob(f"{folder}weather_train.csv")
data = pd.read_csv(files[0])
data["timestamp"] = pd.to_datetime(data.timestamp)
#write(f"Full dataset: {data.shape}")
print("Processing dataset...")


array_list = []

for chosen_building in range(0,1448+1):
    #print(f"\nBuilding ID: {chosen_building}")
    chosen_site = meta_data.loc[meta_data.building_id == chosen_building, "site_id"].values[0]
    #print(f"Site ID: {chosen_site}")
    year_built = meta_data.loc[meta_data.building_id == chosen_building, "year_built"].values[0]
    sq_ft = meta_data.loc[meta_data.building_id == chosen_building, "square_feet"].values[0]

    site_weather = data.loc[data.site_id == chosen_site]

    site_weather = fix_time_gaps(site_weather, start=start, end=end)

    weather_array = site_weather.drop(["site_id","timestamp"], axis=1)
    weather_array = wind_direction_trigonometry(weather_array)
    weather_array = weather_array.drop("cloud_coverage", axis=1)
    weather_array = weather_array.drop("precip_depth_1_hr", axis=1)
    weather_array = weather_array.drop("sea_level_pressure", axis=1)

    #print(f"NaN count: {nan_count_total(weather_array)}")
    #print("Interpolating NaN values...")
    weather_array = nan_mean_interpolation(weather_array)
    #print(f"NaN count: {nan_count_total(weather_array)}")
    nan_count = nan_count_total(weather_array)
    if nan_count > 0:
        print(f"NaN count is {nan_count} at building: {chosen_building}")
    weather_variables = weather_array.columns

    year_column = [year_built] * weather_array.shape[0]
    year_column = pd.Series(year_column)
    weather_array.insert(loc=weather_array.shape[-1], column="year_built", value=year_column)
    average_year = meta_data.year_built.mean()
    weather_array.year_built = weather_array.year_built.fillna(average_year)
    if nan_count > 0:
        print(f"NaN count (year built) is {nan_count} at building: {chosen_building}")

    sqft_column = [sq_ft] * weather_array.shape[0]
    sqft_column = pd.Series(sqft_column)
    weather_array.insert(loc=weather_array.shape[-1], column="square_feet", value=sqft_column)
    average_sqft = meta_data.square_feet.mean()
    weather_array.square_feet = weather_array.square_feet.fillna(average_sqft)
    if nan_count > 0:
        print(f"NaN count (square feet) is {nan_count} at building: {chosen_building}")

    weather_array = weather_array.to_numpy()
    array_list.append(weather_array)
    #print(len(array_list), array_list[-1].shape,"\n")

all_sites_weather = np.vstack(array_list)

print(all_sites_weather.shape)
if all_sites_weather.shape[0] == 12728016:
    print("Successfully stacked weather for all buildings.")


folder = "/space/mwlw3/GTC_data_exploration/data_cleaned/"
#folder = "C:\\Users\\Michelle\\PycharmProjects\\GTC\\data_cleaned\\"

now = current_time()
title = f"MLP_log_{now}"

data_type = "buildings"

print("WEATHER TRAINING DATA")
write(title, f"{current_time()}\nWEATHER TRAINING DATA\n")
weather_array = all_sites_weather
write(title, f"All {data_type}. Includes year built and square feet.")
write(title, f"Data shape: {weather_array.shape}")

#weather_variables = ["air_temperature","dew_temperature","wind_speed", "cos_wind_direction", "sin_wind_direction"]

print("Done.")



print("\nBUILDING TRAINING DATA")
write(title, "\nBUILDING TRAINING DATA\n")
print("Reading dataset...")
files = glob.glob(f"{folder}energy_processed*{data_type}.csv")
energy_array = np.genfromtxt(files[0], delimiter=",")
write(title, f"All {data_type}.")
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

print("Applying normalisation...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
y_train = scaler.fit_transform(y_train)
X_test = scaler.transform(X_test)
y_test = scaler.transform(y_test)


write(title, f"\nNormalised the training data and applied the same to the test set.")

print("Fitting multi-layer perceptron regression model...")
model = MLPRegressor(hidden_layer_sizes=(100,100,100,), batch_size=16, verbose=True, max_iter=200)
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

vars = "_yrbt_sqft"
arch = "_100_100_100"
filename = f"/space/mwlw3/GTC_data_exploration/to_run_models/MLP_model{vars}{arch}.sav"
pickle.dump(model, open(filename, "wb"))
np.savetxt(f"/space/mwlw3/GTC_data_exploration/data_train_test/X_train{vars}.csv", X_train, delimiter=",")
np.savetxt(f"/space/mwlw3/GTC_data_exploration/data_train_test/y_train{vars}.csv", y_train, delimiter=",")
np.savetxt(f"/space/mwlw3/GTC_data_exploration/data_train_test/X_test{vars}.csv", X_test, delimiter=",")
np.savetxt(f"/space/mwlw3/GTC_data_exploration/data_train_test/y_test{vars}.csv", y_test, delimiter=",")

print("Saved model and train/test files.")
