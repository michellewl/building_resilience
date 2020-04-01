import pandas as pd
import glob
from functions.functions import show_data, nan_mean_interpolation, nan_count_by_variable

windows_os = True

if windows_os:
    code_home_folder = "C:\\Users\\Michelle\\OneDrive - University of Cambridge\\MRes\\Guided_Team_Challenge\\building_resilience\\"
    raw_folder = f"{code_home_folder}data\\ashrae\\kaggle_provided\\" # raw data
    data_folder = "data\\ashrae\\processed_arrays\\" # where to save the processed data
else:
    code_home_folder = "/home/mwlw3/Documents/Guided_team_challenge/building_resilience/"
    raw_folder = "/space/mwlw3/GTC_data_exploration/data_ashrae_raw/" # raw data
    data_folder = "data/ashrae/processed_arrays/" # where to save the processed data

print("\nWEATHER TRAINING DATA\n")
files = glob.glob(f"{raw_folder}weather_train.csv")
data = show_data(raw_folder,files[0])

print(f"\nFull dataset: {data.shape}")
print(f"\nNaN count: \n{nan_count_by_variable(data)}")

print("\nBUILDING TRAINING DATA\n")
files = glob.glob(f"{raw_folder}train.csv")
data = show_data(raw_folder,files[0])

print(f"\nFull dataset: {data.shape}")
print(f"\nNaN count: \n{nan_count_by_variable(data)}")
print(f"Start date: {data.timestamp.min()}")
print(f"End date: {data.timestamp.max()}")
data["timestamp"] = pd.to_datetime(data.timestamp)
print(f"Years {data.timestamp.dt.year.unique()}")

print(f"Meter types:\n{data.meter.value_counts()}")

print("\nBUILDING META DATA\n")
files = glob.glob(f"{raw_folder}*meta*.csv")
data = show_data(raw_folder,files[0])

print(f"\nFull dataset: {data.shape}")
print(f"\nNaN count: \n{nan_count_by_variable(data)}")

print(f"\nBuilding purpose:\n{data.primary_use.value_counts()}")

