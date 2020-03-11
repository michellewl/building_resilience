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
dataframe_list = []

for chosen_building in range(0,meta_data.shape[0]):
    if chosen_building%50==0:
        print(f"We're on building #{chosen_building}...")
    chosen_site = meta_data.loc[meta_data.building_id == chosen_building, "site_id"].values[0]
    year_built = meta_data.loc[meta_data.building_id == chosen_building, "year_built"].values[0]
    sq_ft = meta_data.loc[meta_data.building_id == chosen_building, "square_feet"].values[0]

    site_weather = data.loc[data.site_id == chosen_site]

    site_weather = fix_time_gaps(site_weather, start=start, end=end)

    weather_array = site_weather.drop("site_id", axis=1)
    weather_array = wind_direction_trigonometry(weather_array)
    weather_array = weather_array.drop(["cloud_coverage", "precip_depth_1_hr", "sea_level_pressure"], axis=1)

    weather_array.iloc[:, 1:] = nan_mean_interpolation(weather_array.iloc[:, 1:])
    nan_count = nan_count_total(weather_array)
    if nan_count > 0:
        print(f"NaN count is {nan_count} at building: {chosen_building}")
    
    #_______________________squash to daily data
    # All variables
    new_variables = ["mean_air_temp", "mean_dew_temp", "mean_wind_speed", "mean_cos_wind_dir", "mean_sin_wind_dir"]
    daily_weather = weather_array.copy().resample("D", on="timestamp").mean()
    daily_weather.columns = new_variables
    daily_weather = daily_weather.join(weather_array.copy().resample("D", on="timestamp").min().iloc[:,1:])
    new_variables.extend(["min_air_temp", "min_dew_temp", "min_wind_speed", "min_cos_wind_dir", "min_sin_wind_dir"])
    daily_weather.columns = new_variables
    daily_weather = daily_weather.join(weather_array.copy().resample("D", on="timestamp").max().iloc[:,1:])
    new_variables.extend(["max_air_temp", "max_dew_temp", "max_wind_speed", "max_cos_wind_dir", "max_sin_wind_dir"])
    daily_weather.columns = new_variables

    # Additional temperature metrics
    temperature_array = weather_array.iloc[:,:2].copy().set_index("timestamp")
    # temperature thresholds
    daily_weather["hours_above_18_5_degc"] = temperature_array.copy().resample("D")["air_temperature"].apply(lambda x: (x>18.5).sum())
    daily_weather["hours_below_18_5_degc"] = temperature_array.copy().resample("D")["air_temperature"].apply(lambda x: (x<18.5).sum())
    daily_weather["hours_above_25_degc"] = temperature_array.copy().resample("D")["air_temperature"].apply(lambda x: (x>25).sum())
    daily_weather["hours_below_15_5_degc"] = temperature_array.copy().resample("D")["air_temperature"].apply(lambda x: (x<15.5).sum())



    #_______________________squash to monthly data
    monthly_weather = daily_weather.copy().resample("M").mean()
    monthly_weather.insert(loc=monthly_weather.shape[-1], column="number_of_days_per_month", value=monthly_weather.index.day)
    monthly_weather["three_month_average_temp"] = monthly_weather.mean_air_temp.rolling(window=3).mean()
    # Fill NaN gaps for January & February (only have 1 year of data)
    monthly_weather.iloc[0,-1] = monthly_weather.mean_air_temp.values[0]
    monthly_weather.iloc[1,-1] = monthly_weather.mean_air_temp.values[0:2].mean()
    

    #________________________Add in the meta data about each building

    if include_meta_data is True:
        #daily_weather.insert(loc=daily_weather.shape[-1], column="year_built", value=pd.Series([year_built] * daily_weather.shape[0]))
        monthly_weather['year_built'] = [year_built] * monthly_weather.shape[0]
        average_year = meta_data.year_built.mean()
        #daily_weather.year_built = daily_weather.year_built.fillna(average_year)
        monthly_weather.year_built = monthly_weather.year_built.fillna(average_year)
        if nan_count > 0:
            print(f"NaN count (year built) is {nan_count} at building: {chosen_building}")


        #daily_weather.insert(loc=daily_weather.shape[-1], column="square_feet", value=pd.Series([sq_ft] * daily_weather.shape[0]))
        monthly_weather['square_feet'] = [sq_ft] * monthly_weather.shape[0]
        average_sqft = meta_data.square_feet.mean()
        #daily_weather.square_feet = daily_weather.square_feet.fillna(average_sqft)
        monthly_weather.square_feet = monthly_weather.square_feet.fillna(average_sqft)
        if nan_count > 0:
            print(f"NaN count (square feet) is {nan_count} at building: {chosen_building}")

        #daily_weather.insert(loc=daily_weather.shape[-1], column="square_feet", value=pd.Series([sq_ft] * daily_weather.shape[0]))
        monthly_weather['site_id'] = [chosen_site] * monthly_weather.shape[0]
        if nan_count > 0:
            print(f"NaN count (site_id) is {nan_count} at building: {chosen_building}")


    array_list.append(monthly_weather.to_numpy())
    dataframe_list.append(monthly_weather)

all_sites_weather = np.vstack(array_list)
weather_dataframe = pd.concat(dataframe_list)
print(all_sites_weather.shape)


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

for chosen_building in range(0, meta_data.shape[0]):
    if chosen_building%50==0:
        print(f"We're on building #{chosen_building}...")
    
    building = data.loc[data.building_id == chosen_building].copy()

   

    building = building.groupby("timestamp", as_index=False).sum()
    # This adds meter readings together if there multiple energy meters.

    building = fix_time_gaps(building, start=start, end=end)

    building_array = building.meter_reading
    building.meter_reading = nan_mean_interpolation(building.meter_reading)
    nan_count = nan_count_total(building.meter_reading)
    if nan_count > 0:
        print(f"NaN count is {nan_count} at building: {chosen_building}\n{building.head}")

    daily_energy = building.copy().resample("D", on="timestamp").sum()
    monthly_energy = building.copy().resample("M", on="timestamp").mean()
    monthly_energy = monthly_energy.drop("meter_reading", axis=1)
    monthly_energy["mean_daily_energy"] = daily_energy.copy().resample("M").mean().meter_reading
    monthly_energy["total_energy"] = building.copy().resample("M", on="timestamp").sum().meter_reading
    
 
    array_list.append(monthly_energy.total_energy.to_numpy())
    dataframe_list.append(monthly_energy)
    
all_sites_energy = np.concatenate(array_list, axis=None)
energy_dataframe = pd.concat(dataframe_list)
print(all_sites_energy.shape)

if all_sites_energy.shape[0] == all_sites_weather.shape[0]:
    print("\nSuccess!")
else:
    print(f"Error occurred, weather array shape is {all_sites_weather.shape[0]} but energy array shape is {all_sites_energy.shape[0]}.")


save_folder = "data/processed_arrays/"

if include_meta_data is True:
    np.savetxt(f"{code_home_folder}{save_folder}monthly_weather.csv", all_sites_weather, delimiter=",")
    weather_dataframe.to_csv(f"{code_home_folder}{save_folder}monthly_weather_dataframe.csv", index=True)
elif include_meta_data is False:
    np.savetxt(f"{code_home_folder}{save_folder}monthly_weather_nometa.csv", all_sites_weather, delimiter=",")
    weather_dataframe.to_csv(f"{code_home_folder}{save_folder}monthly_weather_dataframe_nometa.csv", index=True)

np.savetxt(f"{code_home_folder}{save_folder}monthly_energy.csv", all_sites_energy, delimiter=",")
energy_dataframe.to_csv(f"{code_home_folder}{save_folder}monthly_energy_dataframe.csv", index=True)
print("\n Successfully saved data files for weather and energy.")

print(monthly_weather.head)
print(monthly_energy.head)
