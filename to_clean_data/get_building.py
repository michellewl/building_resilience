import pandas as pd
import numpy as np
import glob
from functions import show_data
from functions import nan_mean_interpolation, nan_count_by_variable

folder = "/space/mwlw3/GTC/ashrae-energy-prediction/"
#folder = "C:\\Users\\Michelle\\OneDrive\\Documents\\Uni\\MRes\\Guided_Team_Challenge\\data\\Kaggle\\ashrae-energy-prediction\\"

print("BUILDING META DATA")
files = glob.glob(f"{folder}*meta*.csv")
data = pd.read_csv(files[0])

chosen_site = 0
site_buildings = data.loc[data.site_id == chosen_site]
site_buildings = site_buildings.building_id.to_numpy()
print(site_buildings)

def get_building_ids(site_number, meta_data_file):
    data = pd.read_csv(meta_data_file)
    site_buildings = data.loc[data.site_id == site_number]
    site_buildings = site_buildings.building_id.to_numpy()
    return site_buildings
