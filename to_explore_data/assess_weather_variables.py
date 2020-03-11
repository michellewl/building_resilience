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
from functions import nan_mean_interpolation, nan_count_total, nan_count_by_variable, \
    write, get_building_ids, fix_time_gaps, wind_direction_trigonometry

folder = "/space/mwlw3/GTC_data_exploration/ashrae-energy-prediction/"
#folder = "C:\\Users\\Michelle\\PycharmProjects\\GTC\\data\\ashrae-energy-prediction\\"
#chosen_site = 0
#chosen_building = 0

print("WEATHER TRAINING DATA")
#write("WEATHER TRAINING DATA\n")
print("Reading dataset...")
files = glob.glob(f"{folder}weather_train.csv")
data = pd.read_csv(files[0])
print("Processing dataset...")
data["timestamp"] = pd.to_datetime(data.timestamp)
#write(f"Full dataset: {data.shape}")

start = data.timestamp.min()
end = data.timestamp.max()

for chosen_site in range(data.site_id.max()+1):
    print(f"\nSite ID {chosen_site}\n")

    site_weather = data.loc[data.site_id == chosen_site]

    print(site_weather.shape)
    print("Filling time gaps...")
    site_weather = fix_time_gaps(site_weather, start=start, end=end)

    print(site_weather.shape)

    print(f"NaN count: \n{nan_count_by_variable(site_weather)}")

data = wind_direction_trigonometry(data)

data = data.fillna(-40)

variables = list(data.columns)
variables.remove("site_id")
variables.remove("timestamp")
print(variables)

fig, axes = plt.subplots(len(variables), figsize=(20,40))

for i in range(len(variables)):
    axes[i].hist(data[f"{variables[i]}"], bins=30)
    axes[i].set_title(f"{variables[i]}")
plt.show()