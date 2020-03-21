import pandas as pd
import numpy as np
import glob
from functions.functions import nan_mean_interpolation, nan_count_total, nan_count_by_variable, write, \
    get_building_ids, fix_time_gaps, wind_direction_trigonometry
import datetime as dt

windows_os = True

if windows_os:
    code_home_folder = "C:\\Users\\Michelle\\OneDrive - University of Cambridge\\MRes\\Guided_Team_Challenge\\building_resilience\\"
    raw_folder = f"{code_home_folder}data\\ashrae-energy-prediction\\kaggle_provided\\" # raw data
    data_folder = "data\\processed_arrays\\" # where to save the processed data
else:
    code_home_folder = "/home/mwlw3/Documents/Guided_team_challenge/building_resilience/"
    raw_folder = "/space/mwlw3/GTC_data_exploration/data_ashrae_raw/" # raw data
    data_folder = "data/processed_arrays/" # where to save the processed data


include_meta_data = True
monthly_data = False
timely = "daily"

print("\nBUILDING META DATA\n")
files = glob.glob(f"{raw_folder}*meta*.csv")
meta_data = pd.read_csv(files[0])

start = dt.datetime(day=1, month=1, year=2016, hour=0, minute=0)
end = dt.datetime(day=31, month=12, year=2016, hour=23, minute=0)


print("WEATHER TRAINING DATA")
print("Reading dataset...")
files = glob.glob(f"{raw_folder}weather_train.csv")
data = pd.read_csv(files[0])
data["timestamp"] = pd.to_datetime(data.timestamp)
print("Processing dataset...")


dataframe_list= []
# if monthly_data is True:
#     dataframe_list_monthly = []

for chosen_building in range(0, meta_data.shape[0]):
    if chosen_building%50==0:
        print(f"We're on building #{chosen_building}...")
    chosen_site = meta_data.loc[meta_data.building_id == chosen_building, "site_id"].values[0]

    # removing dodgy sites
    if chosen_site == 7 or chosen_site == 9:
        continue


    year_built = meta_data.loc[meta_data.building_id == chosen_building, "year_built"].values[0]
    sq_ft = meta_data.loc[meta_data.building_id == chosen_building, "square_feet"].values[0]
    primary_use = meta_data.loc[meta_data.building_id == chosen_building, "primary_use"].values[0]

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
    daily_weather = daily_weather.join(weather_array.copy().resample("D", on="timestamp").min().iloc[:,1:-2])
    new_variables.extend(["min_air_temp", "min_dew_temp", "min_wind_speed"])
    daily_weather.columns = new_variables
    daily_weather = daily_weather.join(weather_array.copy().resample("D", on="timestamp").max().iloc[:,1:-2])
    new_variables.extend(["max_air_temp", "max_dew_temp", "max_wind_speed"])
    daily_weather.columns = new_variables

    # Additional temperature metrics
    temperature_array = weather_array.iloc[:,:2].copy().set_index("timestamp")
    # temperature thresholds
    daily_weather["hours_above_18_5_degc"] = temperature_array.copy().resample("D")["air_temperature"].apply(lambda x: (x>18.5).sum())
    daily_weather["hours_below_18_5_degc"] = temperature_array.copy().resample("D")["air_temperature"].apply(lambda x: (x<18.5).sum())
    daily_weather["hours_above_25_degc"] = temperature_array.copy().resample("D")["air_temperature"].apply(lambda x: (x>25).sum())
    daily_weather["hours_below_15_5_degc"] = temperature_array.copy().resample("D")["air_temperature"].apply(lambda x: (x<15.5).sum())
    daily_weather["28_day_average_temp"] = daily_weather.mean_air_temp.rolling(window=28).mean()
    weather = daily_weather.copy()

    if monthly_data is True:
        #_______________________squash to monthly data
        weather = daily_weather.copy().resample("M").mean()
        weather.insert(loc=weather.shape[-1], column="number_of_days_per_month", value=weather.index.day)
        weather["three_month_average_temp"] = weather.mean_air_temp.rolling(window=3).mean()
        # Fill NaN gaps for January & February (only have 1 year of data)
        weather.iloc[0,-1] = weather.mean_air_temp.values[0]
        weather.iloc[1,-1] = weather.mean_air_temp.values[0:2].mean()
    

    #________________________Add in the meta data about each building

    if include_meta_data is True:
        weather['year_built'] = [year_built] * weather.shape[0]
        average_year = meta_data.year_built.mean()
        weather.year_built = weather.year_built.fillna(average_year)
                
        weather['square_feet'] = [sq_ft] * weather.shape[0]
        average_sqft = meta_data.square_feet.mean()
        weather.square_feet = weather.square_feet.fillna(average_sqft)
                
        weather['site_id'] = [chosen_site] * weather.shape[0]

        weather['primary_use'] = [primary_use] * weather.shape[0]
        
        # if monthly_data is True:
        #     monthly_weather['year_built'] = [year_built] * monthly_weather.shape[0]
        #     monthly_weather.year_built = monthly_weather.year_built.fillna(average_year)
        #     monthly_weather['square_feet'] = [sq_ft] * monthly_weather.shape[0]
        #     monthly_weather.square_feet = monthly_weather.square_feet.fillna(average_sqft)
        #     monthly_weather['site_id'] = [chosen_site] * monthly_weather.shape[0]

    # include the building ID for joining dataframes later
    weather['building_id'] = [chosen_building] * weather.shape[0]
    weather = weather.reset_index()
    weather = weather.set_index(keys = ["timestamp", "building_id"])


    dataframe_list.append(weather)
    # if monthly_data is True:
    #     dataframe_list_monthly.append(monthly_weather)


weather_dataframe = pd.concat(dataframe_list)

# if monthly_data is True:
#     weather_dataframe_monthly = pd.concat(dataframe_list_monthly)

print("\nBUILDING TRAINING DATA")
print("Reading dataset...")
files = glob.glob(f"{raw_folder}train.csv")
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

dataframe_list= []
# if monthly_data is True:
#     dataframe_list_monthly = []

for chosen_building in range(0, meta_data.shape[0]):
    if chosen_building%50==0:
        print(f"We're on building #{chosen_building}...")

    # removing dodgy sites
    chosen_site = meta_data.loc[meta_data.building_id == chosen_building, "site_id"].values[0]
    if chosen_site is 7 or chosen_site == 9:
        continue

    building = data.loc[data.building_id == chosen_building].copy()

    building = building.loc[building.meter ==0,:]
    building.meter_reading = building.meter_reading* 0.000293071
    # This retains only the electricity meter and converts from kBTU to kWh



    if all(np.isnan(building.meter_reading)) is True:
        #print(f"Skipped building #{chosen_building}.")
        continue
    building = fix_time_gaps(building, start=start, end=end)

  

    building.meter_reading = nan_mean_interpolation(building.meter_reading)
    nan_count = nan_count_total(building.meter_reading)
    if nan_count > 0:
        print(f"NaN count meter_reading is {nan_count} at building: {chosen_building}\n{building.head}")

    

    daily_energy = building.copy().resample("D", on="timestamp").mean()
    daily_energy.meter_reading = building.copy().resample("D", on="timestamp").sum().meter_reading
    energy = daily_energy.copy()
 

    if monthly_data is True:
        energy = building.copy().resample("M", on="timestamp").mean()
        energy = energy.drop("meter_reading", axis=1)
        energy["mean_daily_energy"] = daily_energy.copy().resample("M").mean().meter_reading
        energy["total_energy"] = building.copy().resample("M", on="timestamp").sum().meter_reading
  
    energy = energy.reset_index()
    energy = energy.set_index(keys = ["timestamp", "building_id"])
 
    dataframe_list.append(energy)
    # if monthly_data is True:
    #     dataframe_list_monthly.append(monthly_energy)


energy_dataframe = pd.concat(dataframe_list)

# if monthly_data is True:
#     energy_dataframe_monthly = pd.concat(dataframe_list_monthly)

print(f"Weather array shape is {weather_dataframe.shape[0]} and energy array shape is {energy_dataframe.shape[0]}.")

# if monthly_data is True:
#     print(f"Weather array shape is {weather_dataframe_monthly.shape[0]} and energy array shape is {energy_dataframe_monthly.shape[0]}.")



full_dataframe= weather_dataframe.join(energy_dataframe, how="left")




full_dataframe = full_dataframe.dropna(axis=0)
full_dataframe['electricity_per_sqft'] = full_dataframe['meter_reading'] / full_dataframe['square_feet']

full_dataframe["mean_RH"] = (full_dataframe.mean_dew_temp - full_dataframe.mean_air_temp + 20) * 5
full_dataframe = full_dataframe.drop(["meter_reading", "mean_dew_temp"], axis=1)


if include_meta_data is True:
    full_dataframe.to_csv(f"{code_home_folder}{data_folder}full_dataframe{timely}.csv", index=True)
elif include_meta_data is False:
    full_dataframe.to_csv(f"{code_home_folder}{data_folder}full_dataframe_nometa{timely}.csv", index=True)


print("\n Successfully saved full dataframe.")

print(full_dataframe)