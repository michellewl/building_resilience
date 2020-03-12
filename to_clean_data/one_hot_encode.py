import pandas as pd
import numpy as np
import glob
from functions.functions import nan_mean_interpolation, nan_count_total, nan_count_by_variable, write, \
    get_building_ids, fix_time_gaps, wind_direction_trigonometry
import datetime as dt

raw_folder = "/space/mwlw3/GTC_data_exploration/data_ashrae_raw/"
code_home_folder = "/home/mwlw3/Documents/Guided_team_challenge/building_resilience/"
save_folder = "data/processed_arrays/"


print("\nBUILDING META DATA\n")
files = glob.glob(f"{raw_folder}*meta*.csv")
meta_data = pd.read_csv(files[0])

print("\nFULL DATASET\n")
print("Reading dataset...")
files = glob.glob(f"{code_home_folder}{save_folder}full_dataframe_daily.csv")
data = pd.read_csv(files[0])
data["timestamp"] = pd.to_datetime(data.timestamp)
print("Processing dataset...")


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

print(f"Before one-hot encoding:\n{data.columns}")

hot_data = one_hot(data, "site_id")

sites = list(set(data.site_id.values))
site_dict = {}
for i in sites:
    site_dict[i] = f"site_{i}"

hot_data = hot_data.rename(mapper=site_dict, axis=1)

print(f"After one-hot encoding:\n{hot_data.columns}")

