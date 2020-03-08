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
    get_building_ids, fix_time_gaps, wind_direction_trigonometry
import datetime as dt

folder = "/space/mwlw3/GTC_data_exploration/ashrae-energy-prediction/"
#folder = "C:\\Users\\Michelle\\PycharmProjects\\GTC\\data\\ashrae-energy-prediction\\"
chosen_site = 15
#chosen_building = 0

print("WEATHER TRAINING DATA")
write("WEATHER TRAINING DATA\n")
print("Reading dataset...")
files = glob.glob(f"{folder}weather_train.csv")
data = pd.read_csv(files[0])
data["timestamp"] = pd.to_datetime(data.timestamp)
write(f"Full dataset: {data.shape}")
print("Processing dataset...")
#data = fix_time_gaps(data)
site_weather = data.loc[data.site_id == chosen_site]

start = dt.datetime(day=1, month=1, year=2016, hour=0, minute=0)
end = dt.datetime(day=31, month=12, year=2016, hour=23, minute=0)
site_weather = fix_time_gaps(site_weather, start=start, end=end)

write(f"Subset by site: {site_weather.shape}")
weather_array = site_weather.drop(["site_id","timestamp"], axis=1)

#print("Changing wind direction feature...")
weather_array = wind_direction_trigonometry(weather_array)

print("Removing variables...")
weather_array = weather_array.drop("cloud_coverage", axis=1)
weather_array = weather_array.drop("precip_depth_1_hr", axis=1)
# weather_array = weather_array.drop("sea_level_pressure", axis=1)
# write("Sea Level Pressure variable missing all data.")

print(f"NaN count: {nan_count_total(weather_array)}")
print("Interpolating NaN values...")
weather_array = nan_mean_interpolation(weather_array)
print(f"NaN count: {nan_count_total(weather_array)}")
weather_variables = weather_array.columns

weather_array = weather_array.to_numpy()
write(f"Extracted data columns: {weather_array.shape}")
print("Done.")



print("\nBUILDING TRAINING DATA")
write("\nBUILDING TRAINING DATA\n")
print("Reading dataset...")
files = glob.glob(f"{folder}train.csv")
data = pd.read_csv(files[0])
data["timestamp"] = pd.to_datetime(data.timestamp)
write(f"Full dataset: {data.shape}")
print("Processing dataset...")

meta_data_file = glob.glob(f"{folder}*meta*.csv")[0]
building_ids = get_building_ids(chosen_site, meta_data_file)
building = data.loc[data.building_id.isin(building_ids)].copy()
write(f"Subset by site ({chosen_site}): {building.shape}")

#building = data.loc[data.building_id == chosen_building]
#write(f"Subset by building: {building.shape}")

data_retention = 0.999
top = 1 - (1-data_retention)/2
bottom = (1-data_retention)/2
q_high = building.meter_reading.quantile(top)
q_low = building.meter_reading.quantile(bottom)
write(f"Threshold: {q_high}, {q_low}")
building.loc[building.meter_reading >= q_high, "meter_reading"] = None
building.loc[building.meter_reading <= q_low, "meter_reading"] = None
values_changed = nan_count_total(building)
write(f"Outlier removal: {values_changed} values changed.")

building = building.groupby("timestamp", as_index=False).mean()
write(f"Data averaged (mean) across all buildings for site {chosen_site}: {building.shape}")

building = fix_time_gaps(building, start=start, end=end)
print(building.shape)

building_array = building.meter_reading
print(f"NaN count: {nan_count_total(building_array)}")
building_array = nan_mean_interpolation(building_array)
print(f"NaN count: {nan_count_total(building_array)}")

building_array = building_array.to_numpy()
write(f"Extracted data columns: {building_array.shape}")
print("Done.")

print("\nMODEL TRAIN")
write("\nMODEL TRAIN\n")
X = weather_array
y = building_array

print("Splitting into train and test sets...")
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
write(f"Training size: {1-test_size}")
write(f"Test size: {test_size}")

write(f"Training array dimensions: {X_train.shape} {y_train.shape}")
write(f"Test array dimensions: {X_test.shape} {y_test.shape}")

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

write("\nNormalised the training data and applied the same to the test set.")

print("Fitting linear regression model...")
model = LinearRegression(fit_intercept = False).fit(X_train,y_train)
print(f"R\u00b2: {model.score(X_train,y_train)}")
write("\nLinear regression model fit:")
write(f"R\u00b2: {model.score(X_train,y_train)}")

y_train_predicted = model.predict(X_train)
print(f"Mean squared error: {mean_squared_error(y_train, y_train_predicted)}")
write(f"Mean squared error: {mean_squared_error(y_train, y_train_predicted)}")

print("\nMODEL TEST")
write("\nLinear regression model test:")
print("Predicting the building results...")

y_predicted = model.predict(X_test)

# print(f"Dimensions of predicted building meter readings: {y_predicted.shape}")
# print(f"Dimensions of observed building meter readings: {y_test.shape}")

print(f"Mean squared error: {mean_squared_error(y_test, y_predicted)}")
write(f"Mean squared error: {mean_squared_error(y_test, y_predicted)}")
print("Model coefficients:")
write("\nModel coefficients:")
coefficients = model.coef_.reshape(-1,1).tolist()
for i in range(len(weather_variables)):
    print(f"{weather_variables[i]} : {coefficients[i]}")
    write(f"{weather_variables[i]} : {coefficients[i]}")

print("Model intercept:")
print(model.intercept_)
write("\nModel intercept:")
write(model.intercept_)