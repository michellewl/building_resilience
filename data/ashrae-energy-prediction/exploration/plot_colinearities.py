import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import numpy as np
import glob
from sklearn.linear_model import LinearRegression
from functions import show_data
from functions import nan_mean_interpolation, nan_count_by_variable

#folder = "/space/mwlw3/GTC/ashrae-energy-prediction/"
#folder = "C:\\Users\\Michelle\\OneDrive\\Documents\\Uni\\MRes\\Guided_Team_Challenge\\data\\Kaggle\\ashrae-energy-prediction\\"
folder = "C:\\Users\\Michelle\\PycharmProjects\\GTC\\data\\ashrae-energy-prediction\\"

chosen_site = 0
chosen_building = 0

print("WEATHER TRAINING DATA")
print("Reading dataset...")
files = glob.glob(f"{folder}weather_train.csv")
data = pd.read_csv(files[0])
print("Processing dataset...")
site_weather = data.loc[data.site_id == chosen_site]
weather_array = site_weather.drop(["site_id","timestamp"], axis=1)
weather_array["cos_wind_direction"] = np.cos(np.deg2rad(weather_array["wind_direction"]))
weather_array["sin_wind_direction"] = np.sin(np.deg2rad(weather_array["wind_direction"]))
weather_array = weather_array.drop("wind_direction", axis=1)
weather_array = nan_mean_interpolation(weather_array)
weather_variables = weather_array.columns
#weather_array = weather_array.to_numpy()
print("Done.")


print("BUILDING TRAINING DATA")
print("Reading dataset...")
files = glob.glob(f"{folder}train.csv")
data = pd.read_csv(files[0])
print("Processing dataset...")
building = data.loc[data.building_id == chosen_building]
building_array = building.meter_reading
#building_array = building_array.to_numpy()
print("Done.")

# figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
#
# plt.scatter(weather_array.air_temperature, weather_array.cloud_coverage)
# plt.xlabel("air_temperature")
# plt.ylabel("dew_temperature")
# plt.show()

fig, axes = plt.subplots(2,2)
axes[0,0].scatter(weather_array.air_temperature, weather_array.cloud_coverage)
axes[0,1].scatter(weather_array.air_temperature, weather_array.dew_temperature)
axes[1,0].scatter(weather_array.air_temperature, weather_array.precip_depth_1_hr)
axes[1,1].scatter(weather_array.air_temperature, weather_array.sea_level_pressure)

plt.show()