import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import glob
from sklearn.linear_model import LinearRegression
from functions.functions import show_data, nan_mean_interpolation, nan_count_by_variable

folder = "/space/mwlw3/GTC_data_exploration/data_ashrae_raw/"
#folder = "C:\\Users\\Michelle\\PycharmProjects\\GTC\\data\\ashrae-energy-prediction\\"

print("\nWEATHER TRAINING DATA\n")
files = glob.glob(f"{folder}weather_train.csv")
data = show_data(folder,files[0])

print(f"\nFull dataset: {data.shape}")
print(f"\nNaN count: \n{nan_count_by_variable(data)}")

# print(data.loc[np.isnan(data.cloud_coverage)&(data.precip_depth_1_hr ==-1)])#["precip_depth_1_hr"].nunique())
# plt.hist(data.loc[np.isnan(data.cloud_coverage)]["precip_depth_1_hr"])


print("\nBUILDING TRAINING DATA\n")
files = glob.glob(f"{folder}train.csv")
data = show_data(folder,files[0])

print(f"\nFull dataset: {data.shape}")
print(f"\nNaN count: \n{nan_count_by_variable(data)}")
print(f"Start date: {data.timestamp.min()}")
print(f"End date: {data.timestamp.max()}")
data["timestamp"] = pd.to_datetime(data.timestamp)
print(f"Years {data.timestamp.dt.year.unique()}")

print(f"Meter types:\n{data.meter.value_counts()}")

print("\nBUILDING META DATA\n")
files = glob.glob(f"{folder}*meta*.csv")
data = show_data(folder,files[0])

print(f"\nFull dataset: {data.shape}")
print(f"\nNaN count: \n{nan_count_by_variable(data)}")

print(f"\nBuilding purpose:\n{data.primary_use.value_counts()}")

# plt.hist(data.year_built.values[~np.isnan(data.year_built)])
# plt.show()
