import pandas as pd
import matplotlib.pyplot as plt
import glob
from functions.functions import nan_mean_interpolation, nan_count_by_variable, get_building_ids
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


windows_os = True

if windows_os:
    code_home_folder = "C:\\Users\\Michelle\\OneDrive - University of Cambridge\\MRes\\Guided_Team_Challenge\\building_resilience\\"
    raw_folder = f"{code_home_folder}data\\ashrae\\kaggle_provided\\" # raw data
else:
    code_home_folder = "/home/mwlw3/Documents/Guided_team_challenge/building_resilience/"
    raw_folder = "/space/mwlw3/GTC_data_exploration/data_ashrae_raw/" # raw data

chosen_site = 0

meta_data_file = glob.glob(f"{raw_folder}*meta*.csv")[0]

print("BUILDING TRAINING DATA")
print("Importing data...")
files = glob.glob(f"{raw_folder}train.csv")
data = pd.read_csv(files[0])

building_ids = get_building_ids(chosen_site, meta_data_file)

df = data.loc[data.building_id.isin(building_ids)].copy()
df["timestamp"] = pd.to_datetime(df.timestamp)

q_high = df.meter_reading.quantile(0.9999)
df.meter_reading = df.meter_reading[df.meter_reading < q_high]

print(df)

print("Hourly average for whole site:")
mean_by_time = df.groupby("timestamp").mean()
mean_by_time = mean_by_time.drop(["building_id", "meter"], axis=1)
print(mean_by_time)

print("Annual average per building:")
mean_by_building = df.groupby("building_id").mean()
mean_by_building = mean_by_building.drop("meter", axis=1)
print(mean_by_building)

plt.hist(mean_by_building.meter_reading)
plt.title(f"Annual average energy usage for site {chosen_site} buildings.")
#plt.legend()
#plt.ylim(0,15000)
plt.show()

