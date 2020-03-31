import pandas as pd
from datetime import datetime as dt
import numpy as np
import re

def show_data(folder,file_name):
    open_data = pd.read_csv(file_name)
    file_name = file_name.replace(folder, "")
    print(f"File name: {file_name}")
    print(f"Variables: {list(open_data.columns)}")
    print("Data:")
    print(f"{open_data.head()}")
    print(f"{open_data.tail()}")
    return open_data

def nan_count_total(dataframe):
    nan_count = dataframe.isnull().sum().sum()
    return nan_count

def nan_count_by_variable(dataframe):
    nan_count = dataframe.isnull().sum()
    return nan_count

def nan_mean_interpolation(dataframe):
    dataframe = dataframe.where(
        cond = pd.notnull(dataframe),
        other = (dataframe.fillna(method="ffill") + dataframe.fillna(method="bfill"))/2
    )
    dataframe = dataframe.where(
        cond = pd.notnull(dataframe),
        other = dataframe.fillna(method="ffill")
    )
    dataframe = dataframe.where(
        cond=pd.notnull(dataframe),
        other=dataframe.fillna(method="bfill")
                               )
    return dataframe

def get_building_ids(site_number, meta_data_file):
    data = pd.read_csv(meta_data_file)
    site_buildings = data.loc[data.site_id == site_number]
    site_buildings = site_buildings.building_id.to_numpy()
    return site_buildings

def current_time():
    current_time = dt.now().strftime("%d-%m-%Y_%H%M")
    return current_time

def write(title, string):
    text = open(f"{title}.txt", "a+")
    text.write(f"{string}\n")
    text.close()

def write_nn(string):
    current_time = dt.now().strftime("%d-%m-%Y_%H%M")
    text = open(f"MLP_log_{current_time}.txt", "a+")
    text.write(f"{string}\n")
    text.close()

def fix_time_gaps(dataframe, start, end, frequency="1H"):
    # start = dataframe.timestamp.min()
    # end = dataframe.timestamp.max()
    names = list(dataframe.columns)
    names.remove("timestamp")
    names.insert(0, "timestamp")
    dates = pd.date_range(start=start, end=end, freq=frequency)
    dataframe = dataframe.set_index('timestamp').reindex(dates).reset_index()
    dataframe.columns = names
    return dataframe

def wind_direction_trigonometry(weather_array):
    weather_array["cos_wind_direction"] = np.cos(np.deg2rad(weather_array["wind_direction"]))
    weather_array["sin_wind_direction"] = np.sin(np.deg2rad(weather_array["wind_direction"]))
    weather_array = weather_array.drop("wind_direction", axis=1)
    return weather_array

def one_hot(df, columns):
    '''
    one-hot encode variables in a specified column
    ----------
    Parameters: 
    df (pandas dataframe)
    columns (list of strings): columns to one hot encode
    -------
    Return:
     One pandas dataframe with the one hot encoded columns 
    '''
    new_df = pd.concat((df, pd.get_dummies(df[columns])), axis=1)
    return new_df.drop(columns, axis=1)


def get_models_by_hidden_layers(models_list, hidden_layers):
    list = []
    for model in models_list:
        nodes = re.compile(r'\d+').findall(model)
        if len(nodes) == hidden_layers:
            list.append(model)
    return list

def start_end_datetime(start_day, start_month, start_year, end_day, end_month, end_year):
    start = dt(day=start_day, month=start_month, year=start_year, hour=0, minute=0)
    end = dt(day=end_day, month=end_month, year=end_year, hour=23, minute=0)
    return start, end

def read_timeseries_data(filename):
    data = pd.read_csv(filename)
    data["timestamp"] = pd.to_datetime(data.timestamp)
    return data

def daily_mean_min_max(weather_array):
    new_variables = ["mean_air_temp", "mean_RH", "mean_wind_speed", "mean_cos_wind_dir", "mean_sin_wind_dir"]
    daily_weather = weather_array.copy().resample("D", on="timestamp").mean()
    daily_weather.columns = new_variables
    daily_weather = daily_weather.join(weather_array.copy().resample("D", on="timestamp").min().iloc[:, 1:-2])
    new_variables.extend(["min_air_temp", "min_RH", "min_wind_speed"])
    daily_weather.columns = new_variables
    daily_weather = daily_weather.join(weather_array.copy().resample("D", on="timestamp").max().iloc[:, 1:-2])
    new_variables.extend(["max_air_temp", "max_RH", "max_wind_speed"])
    daily_weather.columns = new_variables
    return daily_weather

def daily_temp_metrics_additional(daily_weather, weather_array):
    temperature_array = weather_array.iloc[:, :2].copy().set_index("timestamp")
    # temperature thresholds
    daily_weather["hours_above_18_5_degc"] = temperature_array.copy().resample("D")["air_temperature"].apply(
        lambda x: (x > 18.5).sum())
    daily_weather["hours_below_18_5_degc"] = temperature_array.copy().resample("D")["air_temperature"].apply(
        lambda x: (x < 18.5).sum())
    daily_weather["hours_above_25_degc"] = temperature_array.copy().resample("D")["air_temperature"].apply(
        lambda x: (x > 25).sum())
    daily_weather["hours_below_15_5_degc"] = temperature_array.copy().resample("D")["air_temperature"].apply(
        lambda x: (x < 15.5).sum())
    daily_weather["28_day_average_temp"] = daily_weather.mean_air_temp.rolling(window=28).mean()
    return daily_weather

def monthly_metrics(daily_weather):
    weather = daily_weather.copy().resample("M").mean()
    weather.insert(loc=weather.shape[-1], column="number_of_days_per_month", value=weather.index.day)
    weather["three_month_average_temp"] = weather.mean_air_temp.rolling(window=3).mean()
    # Fill NaN gaps for January & February (only have 1 year of data)
    weather.iloc[0, -1] = weather.mean_air_temp.values[0]
    weather.iloc[1, -1] = weather.mean_air_temp.values[0:2].mean()
    return weather

def obtain_meta_data(meta_data, chosen_building):
    year_built = meta_data.loc[meta_data.building_id == chosen_building, "year_built"].values[0]
    sq_ft = meta_data.loc[meta_data.building_id == chosen_building, "square_feet"].values[0]
    primary_use = meta_data.loc[meta_data.building_id == chosen_building, "primary_use"].values[0]
    return year_built, sq_ft, primary_use

def dew_to_rh(weather_array):
    weather_array["RH"] = (weather_array.dew_temperature - weather_array.air_temperature + 20) * 5
    return weather_array

def join_meta_data(weather, meta_data, chosen_building):
    chosen_site = meta_data.loc[meta_data.building_id == chosen_building, "site_id"].values[0]
    year_built = meta_data.loc[meta_data.building_id == chosen_building, "year_built"].values[0]
    sq_ft = meta_data.loc[meta_data.building_id == chosen_building, "square_feet"].values[0]
    primary_use = meta_data.loc[meta_data.building_id == chosen_building, "primary_use"].values[0]

    weather['year_built'] = [year_built] * weather.shape[0]
    # if the year built isn't in the meta data, fill with the average year of the whole dataset
    average_year = meta_data.year_built.mean()
    weather.year_built = weather.year_built.fillna(average_year)

    weather['square_feet'] = [sq_ft] * weather.shape[0]
    # if the sq ft isn't in the meta data, fill with the average sq ft of the whole dataset
    average_sqft = meta_data.square_feet.mean()
    weather.square_feet = weather.square_feet.fillna(average_sqft)

    weather['site_id'] = [chosen_site] * weather.shape[0]

    weather['primary_use'] = [primary_use] * weather.shape[0]
    return weather

def remove_outliers(data, data_retention=0.999):
    top = 1 - (1 - data_retention) / 2
    bottom = (1 - data_retention) / 2
    q_high = data.meter_reading.quantile(top)
    q_low = data.meter_reading.quantile(bottom)
    data.loc[data.meter_reading >= q_high, "meter_reading"] = None
    data.loc[data.meter_reading <= q_low, "meter_reading"] = None
    return data, q_high, q_low

def electricity_conversion(building):
    building = building.loc[building.meter == 0, :]
    building.meter_reading = building.meter_reading * 0.000293071
    return building

def daily_energy_total(building):
    daily_energy = building.copy().resample("D", on="timestamp").mean()
    daily_energy.meter_reading = building.copy().resample("D", on="timestamp").sum().meter_reading
    return daily_energy

def monthly_energy_metrics(building, daily_energy):
    energy = building.copy().resample("M", on="timestamp").mean()
    energy = energy.drop("meter_reading", axis=1)
    energy["mean_daily_energy"] = daily_energy.copy().resample("M").mean().meter_reading
    energy["total_energy"] = building.copy().resample("M", on="timestamp").sum().meter_reading
    return energy

def join_dataframes(weather_dataframe, energy_dataframe):
    full_dataframe = weather_dataframe.join(energy_dataframe, how="left")
    full_dataframe = full_dataframe.dropna(axis=0)
    return full_dataframe

def energy_per_sqft(full_dataframe):
    full_dataframe['electricity_per_sqft'] = full_dataframe['meter_reading'] / full_dataframe['square_feet']
    full_dataframe = full_dataframe.drop("meter_reading", axis=1)
    return full_dataframe

def rename_site_cols(sites, hot_data):
    site_dict = {}
    for i in sites:
        site_dict[i] = f"site_{i}"

    hot_data = hot_data.rename(mapper=site_dict, axis=1)
    return hot_data