import pandas as pd
import numpy as np
from functions import nan_mean_interpolation, nan_count_total, \
    fix_time_gaps, wind_direction_trigonometry, one_hot, start_end_datetime, \
    read_timeseries_data, daily_mean_min_max, daily_temp_metrics_additional, \
    monthly_metrics, obtain_meta_data, dew_to_rh, join_meta_data, remove_outliers, \
    electricity_conversion, daily_energy_total, monthly_energy_metrics, join_dataframes, \
    energy_per_sqft, rename_site_cols


windows_os = True

if windows_os:
    code_home_folder = "C:\\Users\\Michelle\\OneDrive - University of Cambridge\\MRes\\Guided_Team_Challenge\\building_resilience\\"
    raw_folder = f"{code_home_folder}data\\ashrae\\kaggle_provided\\" # raw data
    data_folder = "data\\processed_arrays\\" # where to save the processed data
else:
    code_home_folder = "/home/mwlw3/Documents/Guided_team_challenge/building_resilience/"
    raw_folder = "/space/mwlw3/GTC_data_exploration/data_ashrae_raw/" # raw data
    data_folder = "data/ashrae/processed_arrays/" # where to save the processed data

include_meta_data = True
monthly_data = False
timely = "daily"

print("\nBUILDING META DATA\n")
meta_data = pd.read_csv(f"{raw_folder}*meta*.csv")

start, end = start_end_datetime(1, 1, 2016, 31, 12, 2016)

print("WEATHER TRAINING DATA")
print("Reading dataset...")
data = read_timeseries_data(f"{raw_folder}weather_train.csv")
print("Processing dataset...")

dataframe_list= []

for chosen_building in range(0, meta_data.shape[0]):
    if chosen_building%50==0:
        print(f"We're on building #{chosen_building}...")
    chosen_site = meta_data.loc[meta_data.building_id == chosen_building, "site_id"].values[0]

    # removing sites as identified in data exploration
    if chosen_site == 7 or chosen_site == 9:
        continue

    site_weather = data.loc[data.site_id == chosen_site]
    site_weather = fix_time_gaps(site_weather, start=start, end=end)

    # remove periods identified in data exploration
    if chosen_site == 0:
        site_weather = site_weather.loc[site_weather.timestamp >=
                                        dt.datetime.strptime("2016-06-01 00:00:00", '%Y-%m-%d %H:%M:%S')]
    if chosen_site == 15:
        site_weather_a = site_weather.loc[site_weather.timestamp <
                                        dt.datetime.strptime("2016-02-11 00:00:00", '%Y-%m-%d %H:%M:%S')]
        site_weather = site_weather_a.append(site_weather.loc[site_weather.timestamp >=
                                        dt.datetime.strptime("2016-03-30 00:00:00", '%Y-%m-%d %H:%M:%S')])

    weather_array = site_weather.drop("site_id", axis=1)
    weather_array = wind_direction_trigonometry(weather_array)
    weather_array= dew_to_rh(weather_array)
    weather_array = weather_array.drop(["cloud_coverage", "precip_depth_1_hr", "sea_level_pressure", "dew_temperature"]
                                       , axis=1)
    weather_array.iloc[:, 1:] = nan_mean_interpolation(weather_array.iloc[:, 1:])

    if nan_count_total(weather_array) > 0:
        print(f"NaN count is {nan_count_total(weather_array)} at building: {chosen_building}")

    # squash to daily data
    daily_weather = daily_mean_min_max(weather_array)
    daily_weather = daily_temp_metrics_additional(daily_weather, weather_array)
    weather = daily_weather.copy()

    if monthly_data is True:
        weather = monthly_metrics(daily_weather)

    if include_meta_data is True:
        weather = join_meta_data(weather, meta_data, chosen_building)

    # include the building ID for joining dataframes later
    weather['building_id'] = [chosen_building] * weather.shape[0]
    weather = weather.reset_index().set_index(keys = ["timestamp", "building_id"])
    dataframe_list.append(weather)

weather_dataframe = pd.concat(dataframe_list)

print("\nBUILDING TRAINING DATA")
print("Reading dataset...")
data = read_timeseries_data(f"{raw_folder}train.csv")
print("Processing dataset...")

# process outliers of the whole dataset
data, q_high, q_low = remove_outliers(data, data_retention=0.999)
print(f"Outlier limits: {q_low}, {q_high}")

dataframe_list= []

for chosen_building in range(0, meta_data.shape[0]):
    if chosen_building%50==0:
        print(f"We're on building #{chosen_building}...")

    chosen_site = meta_data.loc[meta_data.building_id == chosen_building, "site_id"].values[0]
    # removing sites as identified in data exploration
    if chosen_site is 7 or chosen_site == 9:
        continue

    building = data.loc[data.building_id == chosen_building].copy()
    building = electricity_conversion(building)

    # skip building if no electricity data
    if all(np.isnan(building.meter_reading)) is True:
        continue

    building = fix_time_gaps(building, start=start, end=end)
    building.meter_reading = nan_mean_interpolation(building.meter_reading)

    if nan_count_total(building.meter_reading) > 0:
        print(f"NaN count meter_reading is {nan_count_total(building.meter_reading)} at building: {chosen_building}")

    daily_energy = daily_energy_total(building)
    energy = daily_energy.copy()

    if monthly_data is True:
        energy = monthly_energy_metrics(building, daily_energy)

    energy = energy.reset_index()
    energy = energy.set_index(keys = ["timestamp", "building_id"])
    dataframe_list.append(energy)

energy_dataframe = pd.concat(dataframe_list)

print(f"Weather array shape is {weather_dataframe.shape[0]} and energy array shape is {energy_dataframe.shape[0]}.")

full_dataframe = join_dataframes(weather_dataframe, energy_dataframe)
full_dataframe = energy_per_sqft(full_dataframe)

### One-hot encoding
print(f"Before one-hot encoding:\n{data.columns}")
hot_data = one_hot(full_dataframe, "site_id")
hot_data = rename_site_cols(list(set(full_dataframe.site_id.values)), hot_data)
hot_data = one_hot(hot_data, "primary_use")
print(f"After one-hot encoding:\n{hot_data.columns}")

if include_meta_data is True:
    hot_data.to_csv(f"{code_home_folder}{data_folder}full_dataframe_{timely}.csv", index=True)
elif include_meta_data is False:
    hot_data.to_csv(f"{code_home_folder}{data_folder}full_dataframe_nometa_{timely}.csv", index=True)

print("\n Successfully saved full dataframe.")