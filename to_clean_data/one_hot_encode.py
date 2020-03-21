import pandas as pd
import glob
from functions.functions import one_hot

windows_os = True

if windows_os:
    code_home_folder = "C:\\Users\\Michelle\\OneDrive - University of Cambridge\\MRes\\Guided_Team_Challenge\\building_resilience\\"
    raw_folder = f"{code_home_folder}data\\ashrae-energy-prediction\\kaggle_provided\\" # raw data
    data_folder = "data\\processed_arrays\\" # processed data
else:
    code_home_folder = "/home/mwlw3/Documents/Guided_team_challenge/building_resilience/"
    raw_folder = "/space/mwlw3/GTC_data_exploration/data_ashrae_raw/" # raw data
    data_folder = "data/processed_arrays/" # processed data


print("\nBUILDING META DATA\n")
files = glob.glob(f"{raw_folder}*meta*.csv")
meta_data = pd.read_csv(files[0])

print("\nFULL DATASET\n")
print("Reading dataset...")
files = glob.glob(f"{code_home_folder}{data_folder}full_dataframe_daily.csv")
data = pd.read_csv(files[0])
data["timestamp"] = pd.to_datetime(data.timestamp)
print("Processing dataset...")



print(f"Before one-hot encoding:\n{data.columns}")

hot_data = one_hot(data, "site_id")

sites = list(set(data.site_id.values))
site_dict = {}
for i in sites:
    site_dict[i] = f"site_{i}"

hot_data = hot_data.rename(mapper=site_dict, axis=1)
hot_data = one_hot(hot_data, "primary_use")

print(f"After one-hot encoding:\n{hot_data.columns}")

hot_data.to_csv(f"{code_home_folder}{data_folder}full_dataframe_daily.csv", index=True)

print("Saved.")