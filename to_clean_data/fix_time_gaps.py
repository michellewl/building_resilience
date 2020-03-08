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
from functions import nan_mean_interpolation, nan_count_total, nan_count_by_variable, write, get_building_ids
import datetime as dt

#folder = "/space/mwlw3/GTC/ashrae-energy-prediction/"
folder = "C:\\Users\\Michelle\\PycharmProjects\\GTC\\data\\ashrae-energy-prediction\\"
chosen_site = 1
#chosen_building = 0

print("WEATHER TRAINING DATA")
#write("WEATHER TRAINING DATA\n")
print("Reading dataset...")
files = glob.glob(f"{folder}weather_train.csv")
data = pd.read_csv(files[0])
data["timestamp"] = pd.to_datetime(data.timestamp)
#write(f"Full dataset: {data.shape}")
print("Processing dataset...")
site_weather = data.loc[data.site_id == chosen_site]
#write(f"Subset by site: {site_weather.shape}")

times_gaps = site_weather.timestamp - site_weather.timestamp.shift(1)
#print(times_gaps[times_gaps > dt.timedelta(hours=1, minutes=0, seconds=0)])
#print(site_weather[times_gaps > dt.timedelta(hours=1, minutes=0, seconds=0)])
print((times_gaps-dt.timedelta(hours=1)).sum())

print("Getting start and end times...")
start = site_weather.timestamp.min()
end = site_weather.timestamp.max()
names = list(site_weather.columns)
names.remove("timestamp")
names.insert(0, "timestamp")

print("Creating full datetime range...")
dates = pd.date_range(start=start, end=end, freq='1H')

site_weather = site_weather.set_index('timestamp').reindex(dates).reset_index()
site_weather.columns = names

# times_gaps = site_weather.timestamp - site_weather.timestamp.shift(1)
# print((times_gaps-dt.timedelta(hours=1)).sum())

#site_weather = site_weather.drop(["site_id","timestamp"], axis=1)

print(site_weather.shape)
print (site_weather)


