import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import glob
from sklearn.linear_model import LinearRegression
from functions import show_data
from functions import nan_mean_interpolation, nan_count_by_variable

folder = "/space/mwlw3/GTC_data_exploration/ashrae-energy-prediction/"
#folder = "C:\\Users\\Michelle\\PycharmProjects\\GTC\\data\\ashrae-energy-prediction\\"

# print("\nWEATHER TRAINING DATA\n")
# files = glob.glob(f"{folder}weather_train.csv")
# data = pd.read_csv(files[0])

# # print(f"\nFull dataset: {data.shape}")
# # print(f"\nNaN count: \n{nan_count_by_variable(data)}")




# print("\nBUILDING TRAINING DATA\n")
# files = glob.glob(f"{folder}train.csv")
# data = pd.read_csv(files[0])

# print(f"\nFull dataset: {data.shape}")
# print(f"\nNaN count: \n{nan_count_by_variable(data)}")
# print(f"Start date: {data.timestamp.min()}")
# print(f"End date: {data.timestamp.max()}")

print("\nBUILDING META DATA\n")
files = glob.glob(f"{folder}*meta*.csv")
data = pd.read_csv(files[0])

# print(f"\nFull dataset: {data.shape}")
# print(f"\nNaN count: \n{nan_count_by_variable(data)}")

#print(f"\nBuilding purpose:\n{data.primary_use.value_counts()}")


df = data.copy()
df = df.groupby(["site_id"])["primary_use"].value_counts()
print(df)

df.to_csv("building_use_by_site.csv")