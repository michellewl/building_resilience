import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import glob
import datetime as dt

code_home_folder = "/home/mwlw3/Documents/Guided_team_challenge/building_resilience/"
data_folder = "data/processed_arrays/"

folder = "/space/mwlw3/GTC_data_exploration/data_ashrae_raw/"
#folder = "C:\\Users\\Michelle\\PycharmProjects\\GTC\\data\\ashrae-energy-prediction\\"

print("\nBUILDING META DATA\n")
files = glob.glob(f"{folder}*meta*.csv")
meta_data = pd.read_csv(files[0])

start = dt.datetime(day=1, month=1, year=2016, hour=0, minute=0)
end = dt.datetime(day=31, month=12, year=2016, hour=23, minute=0)

print("\nBUILDING TRAINING DATA")
print("Reading dataset...")
files = glob.glob(f"{code_home_folder}{data_folder}*dataframe.csv")
data = pd.read_csv(files[0])
data["timestamp"] = pd.to_datetime(data.timestamp)
print("Processing dataset...")
#meta_data_file = glob.glob(f"{folder}*meta*.csv")[0]

fig, ax = plt.subplots(figsize=(20,8))

for chosen_building in range(0, 1448+1):
    building = data.loc[data.building_id == chosen_building].copy()
    chosen_site = meta_data.loc[meta_data.building_id == chosen_building, "site_id"].values[0]

    # #remove outliers
    # data_retention = 0.9999
    # top = 1 - (1-data_retention)/2
    # bottom = (1-data_retention)/2
    # q_high = building.meter_reading.quantile(top)
    # q_low = building.meter_reading.quantile(bottom)
    # building.loc[building.meter_reading >= q_high, "meter_reading"] = None
    # building.loc[building.meter_reading <= q_low, "meter_reading"] = None

    #building = building.groupby("timestamp", as_index=False).sum()
    # This adds meter readings together if there multiple energy meters.

    # building = fix_time_gaps(building, start=start, end=end)
    # building.meter_reading = nan_mean_interpolation(building.meter_reading)
    ax.plot(building.timestamp, building.meter_reading, color=f"C{chosen_site}", alpha=0.3)

#plt.legend()
#plt.ylim(0,100000)
plt.show()
   
  