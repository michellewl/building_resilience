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

def get_models_by_hidden_layers(models_list, hidden_layers):
    list = []
    for model in models_list:
        nodes = re.compile(r'\d+').findall(model)
        if len(nodes) == hidden_layers:
            list.append(model)
    return list