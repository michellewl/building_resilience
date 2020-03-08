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

#folder = "/space/mwlw3/GTC_data_exploration/ashrae-energy-prediction/"
folder = "C:\\Users\\Michelle\\PycharmProjects\\GTC\\data\\ashrae-energy-prediction\\"

print("WEATHER TRAINING DATA")
#write("WEATHER TRAINING DATA\n")
print("Reading dataset...")
files = glob.glob(f"{folder}weather_train.csv")
data = pd.read_csv(files[0])
data["timestamp"] = pd.to_datetime(data.timestamp)
#write(f"Full dataset: {data.shape}")
print("Processing dataset...")


array_list = []

for chosen_site in range(0,15+1):
    print(f"\nSite ID: {chosen_site}")
    site_weather = data.loc[data.site_id == chosen_site]

    start = dt.datetime(day=1, month=1, year=2016, hour=0, minute=0)
    end = dt.datetime(day=31, month=12, year=2016, hour=23, minute=0)
    site_weather = fix_time_gaps(site_weather, start=start, end=end)

    weather_array = site_weather.drop(["site_id","timestamp"], axis=1)
    weather_array = wind_direction_trigonometry(weather_array)
    weather_array = weather_array.drop("cloud_coverage", axis=1)
    weather_array = weather_array.drop("precip_depth_1_hr", axis=1)
    weather_array = weather_array.drop("sea_level_pressure", axis=1)

    print(f"NaN count: {nan_count_total(weather_array)}")
    print("Interpolating NaN values...")
    weather_array = nan_mean_interpolation(weather_array)
    print(f"NaN count: {nan_count_total(weather_array)}")
    weather_variables = weather_array.columns

    weather_array = weather_array.to_numpy()
   
    array_list.append(weather_array)
    print(len(array_list), array_list[-1].shape,"\n")

all_sites_weather = np.vstack(array_list)

print(all_sites_weather.shape)

print("\nBUILDING TRAINING DATA")
#write("\nBUILDING TRAINING DATA\n")
print("Reading dataset...")
files = glob.glob(f"{folder}train.csv")
data = pd.read_csv(files[0])
data["timestamp"] = pd.to_datetime(data.timestamp)
#write(f"Full dataset: {data.shape}")
print("Processing dataset...")
meta_data_file = glob.glob(f"{folder}*meta*.csv")[0]

array_list = []

for chosen_site in range(0, 15+1):
    print(f"\nSite ID: {chosen_site}")
    building_ids = get_building_ids(chosen_site, meta_data_file)
    building = data.loc[data.building_id.isin(building_ids)].copy()
    #write(f"Subset by site ({chosen_site}): {building.shape}")

    data_retention = 0.999
    top = 1 - (1-data_retention)/2
    bottom = (1-data_retention)/2
    q_high = building.meter_reading.quantile(top)
    q_low = building.meter_reading.quantile(bottom)
    #write(f"Threshold: {q_high}, {q_low}")
    building.loc[building.meter_reading >= q_high, "meter_reading"] = None
    building.loc[building.meter_reading <= q_low, "meter_reading"] = None
    values_changed = nan_count_total(building)
    #write(f"Outlier removal: {values_changed} values changed.")

    building = building.groupby("timestamp", as_index=False).mean()
    #write(f"Data averaged (mean) across all buildings for site {chosen_site}: {building.shape}")

    building = fix_time_gaps(building, start=start, end=end)
    #print(building.shape)

    building_array = building.meter_reading
    print(f"NaN count: {nan_count_total(building_array)}")
    building_array = nan_mean_interpolation(building_array)
    print(f"NaN count: {nan_count_total(building_array)}")

    building_array = building_array.to_numpy()

    array_list.append(building_array)
    print(len(array_list), array_list[-1].shape,"\n")

all_sites_energy = np.concatenate(array_list, axis=None)

print(all_sites_energy.shape)

if all_sites_weather.shape[0] == all_sites_energy.shape[0]:
    print("\nSuccess!")

np.savetxt("weather_processed_stacked_sites.csv", all_sites_weather, delimiter=",")
np.savetxt("energy_processed_stacked_sites.csv", all_sites_energy, delimiter=",")