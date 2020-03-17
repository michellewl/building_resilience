import pandas as pd
import numpy as np
import glob
from functions.functions import nan_mean_interpolation, nan_count_total, nan_count_by_variable, write, \
    get_building_ids, fix_time_gaps, wind_direction_trigonometry
import datetime as dt

folder = "/space/mwlw3/GTC_data_exploration/data_ashrae_raw/"
#folder = "C:\\Users\\Michelle\\PycharmProjects\\GTC\\data_ashrae_raw\\"

code_home_folder = "/home/mwlw3/Documents/Guided_team_challenge/building_resilience/"

include_meta_data = True

print("\nBUILDING META DATA\n")
files = glob.glob(f"{folder}*meta*.csv")
meta_data = pd.read_csv(files[0])

start = dt.datetime(day=1, month=1, year=2016, hour=0, minute=0)
end = dt.datetime(day=31, month=12, year=2016, hour=23, minute=0)


print("WEATHER TRAINING DATA")
print("Reading dataset...")
files = glob.glob(f"{folder}weather_train.csv")
data = pd.read_csv(files[0])
data["timestamp"] = pd.to_datetime(data.timestamp)
print("Processing dataset...")


array_list = []

for chosen_building in range(0,1448+1):
    chosen_site = meta_data.loc[meta_data.building_id == chosen_building, "site_id"].values[0]
    year_built = meta_data.loc[meta_data.building_id == chosen_building, "year_built"].values[0]
    sq_ft = meta_data.loc[meta_data.building_id == chosen_building, "square_feet"].values[0]

    site_weather = data.loc[data.site_id == chosen_site]

    site_weather = fix_time_gaps(site_weather, start=start, end=end)

    weather_array = site_weather.drop(["site_id","timestamp"], axis=1)
    weather_array = wind_direction_trigonometry(weather_array)
    weather_array = weather_array.drop("cloud_coverage", axis=1)
    weather_array = weather_array.drop("precip_depth_1_hr", axis=1)
    weather_array = weather_array.drop("sea_level_pressure", axis=1)

    weather_array = nan_mean_interpolation(weather_array)
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
    
all_sites_weather = np.vstack(array_list)

print(all_sites_weather.shape)
if all_sites_weather.shape[0] == 12728016:
    print("Successfully stacked weather for all buildings.")

print("\nBUILDING TRAINING DATA")
print("Reading dataset...")
files = glob.glob(f"{folder}train.csv")
data = pd.read_csv(files[0])
data["timestamp"] = pd.to_datetime(data.timestamp)
print("Processing dataset...")

data_retention = 0.999
top = 1 - (1-data_retention)/2
bottom = (1-data_retention)/2
q_high = data.meter_reading.quantile(top)
q_low = data.meter_reading.quantile(bottom)
data.loc[data.meter_reading >= q_high, "meter_reading"] = None
data.loc[data.meter_reading <= q_low, "meter_reading"] = None

print(f"Outlier limits: {q_low}, {q_high}")

array_list = []
dataframe_list = []

for chosen_building in range(0, 1448+1):
    building = data.loc[data.building_id == chosen_building].copy()

   

    building = building.groupby("timestamp", as_index=False).sum()
    # This adds meter readings together if there multiple energy meters.

    building = fix_time_gaps(building, start=start, end=end)

    building_array = building.meter_reading
    building.meter_reading = nan_mean_interpolation(building.meter_reading)
    nan_count = nan_count_total(building.meter_reading)
    if nan_count > 0:
        print(f"NaN count is {nan_count} at building: {chosen_building}\n{building.head}")

    building_array = building.meter_reading.to_numpy()

    array_list.append(building_array)
    dataframe_list.append(building)
    
all_sites_energy = np.concatenate(array_list, axis=None)
energy_dataframe = pd.concat(dataframe_list)

if all_sites_energy.shape[0] == 12728016:
    print("\nSuccessfully stacked energy for all buildings!")
else:
    print(f"Error occurred, energy array shape is {all_sites_energy.shape[0]}.")


save_folder = "data/processed_arrays/"

# if include_meta_data is True:
#     np.savetxt(f"{code_home_folder}{save_folder}weather_processed_stacked_buildings.csv", all_sites_weather, delimiter=",")
# elif include_meta_data is False:
#     np.savetxt(f"{code_home_folder}{save_folder}weather_only_processed_stacked_buildings.csv", all_sites_weather, delimiter=",")

#np.savetxt(f"{code_home_folder}{save_folder}energy_processed_stacked_buildings.csv", all_sites_energy, delimiter=",")
energy_dataframe.to_csv(f"{code_home_folder}{save_folder}energy_stacked_buildings_dataframe.csv", index=False)
print("\n Successfully saved data files for weather and energy.")
