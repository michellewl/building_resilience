import pandas as pd
import matplotlib.pyplot as plt
import glob
from functions import nan_mean_interpolation, nan_count_by_variable, get_building_ids
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


folder = "/space/mwlw3/GTC_data_exploration/ashrae-energy-prediction/"
#folder = "C:\\Users\\Michelle\\PycharmProjects\\GTC\\data\\ashrae-energy-prediction\\"

chosen_site = 0

meta_data_file = glob.glob(f"{folder}*meta*.csv")[0]

print("BUILDING TRAINING DATA")
files = glob.glob(f"{folder}train.csv")
data = pd.read_csv(files[0])

print("Data imported.")

building_ids = get_building_ids(chosen_site, meta_data_file)
#print(f"Building subset: {building_ids}")


df = data.loc[data.building_id.isin(building_ids)].copy()
df["timestamp"] = pd.to_datetime(df.timestamp)

q_high = df.meter_reading.quantile(0.9999)
df.meter_reading = df.meter_reading[df.meter_reading < q_high]

print(df)
# figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
# plt.scatter(df.timestamp,df.meter_reading)
# plt.show()

# one_hot_encoded_id = pd.get_dummies(df.building_id)
# print(one_hot_encoded_id.shape)
# print(df.shape)

#mean_by_building 
#print(df.meter_reading.mean())

print("Average across buildings:")
mean_by_time = df.groupby("timestamp").mean()
mean_by_time = mean_by_time.drop(["building_id", "meter"], axis=1)
print(mean_by_time)

print("Average across time:")
mean_by_building = df.groupby("building_id").mean()
mean_by_building = mean_by_building.drop("meter", axis=1)
print(mean_by_building)


plt.hist(mean_by_building.meter_reading)
#plt.legend()
#plt.ylim(0,15000)
plt.show()

