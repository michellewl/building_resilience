import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

windows_os = True

if windows_os:
    code_home_folder = "C:\\Users\\Michelle\\OneDrive - University of Cambridge\\MRes\\Guided_Team_Challenge\\building_resilience\\"
    raw_folder = f"{code_home_folder}data\\ashrae-energy-prediction\\kaggle_provided\\" # raw data
    data_folder = "data\\ashrae\\processed_arrays\\" # processed data
    save_folder = "data\\ashrae\\train_test_arrays\\"
else:
    code_home_folder = "/home/mwlw3/Documents/Guided_team_challenge/building_resilience/"
    raw_folder = "/space/mwlw3/GTC_data_exploration/data_ashrae_raw/" # raw data
    data_folder = "data/ashrae/processed_arrays/" # processed data
    save_folder = "data/ashrae/train_test_arrays/" # where to save the new arrays

print("\nFULL DATASET\n")
print("Reading dataset...")
files = glob.glob(f"{code_home_folder}{data_folder}full_dataframe_daily.csv")
data = pd.read_csv(files[0])
data["timestamp"] = pd.to_datetime(data.timestamp)
print("Processing dataset...")

y = data.electricity_per_sqft.to_numpy()
X = data.drop(["timestamp", "building_id", "meter", "electricity_per_sqft"], axis=1).to_numpy()

print(f"X: {X.shape}\ny: {y.shape}")

print("Splitting into train, validation and test sets...")
test_size = 0.15
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

validation_size = 0.2
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=42)

print("Applying normalisation...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

print("Saving train, validation & test files...")

np.save(f"{code_home_folder}{save_folder}X_train.npy", X_train)
np.save(f"{code_home_folder}{save_folder}y_train.npy", y_train)
np.save(f"{code_home_folder}{save_folder}X_test.npy", X_test)
np.save(f"{code_home_folder}{save_folder}y_test.npy", y_test)
np.save(f"{code_home_folder}{save_folder}X_val.npy", X_val)
np.save(f"{code_home_folder}{save_folder}y_val.npy", y_val)

print("Saved.")

