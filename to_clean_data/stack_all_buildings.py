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

include_meta_data = False

print("\nBUILDING META DATA\n")
files = glob.glob(f"{folder}*meta*.csv")
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

    if include_meta_data is True:
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

# print("\nBUILDING TRAINING DATA")
# print("Reading dataset...")
# files = glob.glob(f"{folder}train.csv")
# data = pd.read_csv(files[0])
# data["timestamp"] = pd.to_datetime(data.timestamp)
# print("Processing dataset...")
# meta_data_file = glob.glob(f"{folder}*meta*.csv")[0]

# array_list = []

# for chosen_building in range(0, 1448+1):
#     building = data.loc[data.building_id == chosen_building].copy()

#     data_retention = 0.9999
#     top = 1 - (1-data_retention)/2
#     bottom = (1-data_retention)/2
#     q_high = building.meter_reading.quantile(top)
#     q_low = building.meter_reading.quantile(bottom)
#     building.loc[building.meter_reading >= q_high, "meter_reading"] = None
#     building.loc[building.meter_reading <= q_low, "meter_reading"] = None

#     building = building.groupby("timestamp", as_index=False).sum()
#     # This adds meter readings together if there multiple energy meters.

#     building = fix_time_gaps(building, start=start, end=end)

#     building_array = building.meter_reading
#     #print(f"NaN count: {nan_count_total(building_array)}")
#     building_array = nan_mean_interpolation(building_array)
#     #print(f"NaN count: {nan_count_total(building_array)}")
#     nan_count = nan_count_total(building_array)
#     if nan_count > 0:
#         print(f"NaN count is {nan_count} at building: {chosen_building}\n{building.head}")

#     building_array = building_array.to_numpy()

#     array_list.append(building_array)
#     #print(len(array_list), array_list[-1].shape,"\n")

# all_sites_energy = np.concatenate(array_list, axis=None)

# print(all_sites_energy.shape)

# if all_sites_weather.shape[0] == all_sites_energy.shape[0]:
#     print("\nSuccessfully stacked energy for all buildings!")

if inclue_meta_data is True:
    np.savetxt("weather_processed_stacked_buildings.csv", all_sites_weather, delimiter=",")
else:
    np.savetxt("weather_only_processed_stacked_buildings.csv", all_sites_weather, delimiter=",")
# np.savetxt("energy_processed_stacked_buildings.csv", all_sites_energy, delimiter=",")
print("\n Successfully saved data files for weather and energy.")
