import pandas as pd
import numpy as np
import glob
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from functions.functions import write, current_time


code_home_folder = "/home/mwlw3/Documents/Guided_team_challenge/building_resilience/"

title = f"{code_home_folder}logs/training/daily_data/linear_regression_log_{current_time()}"


data_folder = "data/train_test_arrays/"

write(title, f"{current_time()}\nWEATHER TRAINING DATA\n")
write(title, f"All buildings. Daily data, including meta data.")

write(title, "\nBUILDING TRAINING DATA\n")
write(title, f"All buildings, daily total energy.")

print("Importing data...")
X_train = np.genfromtxt(glob.glob(f"{code_home_folder}{data_folder}X_train.csv")[0], delimiter=",")
y_train = np.genfromtxt(glob.glob(f"{code_home_folder}{data_folder}y_train.csv")[0], delimiter=",")
X_test = np.genfromtxt(glob.glob(f"{code_home_folder}{data_folder}X_test.csv")[0], delimiter=",")
y_test = np.genfromtxt(glob.glob(f"{code_home_folder}{data_folder}y_test.csv")[0], delimiter=",")

write(title, f"Training array dimensions: {X_train.shape} {y_train.shape}")
write(title, f"Test array dimensions: {X_test.shape} {y_test.shape}")
# print("WEATHER TRAINING DATA")
# write(title, "WEATHER TRAINING DATA\n")
# print("Reading dataset...")
# files = glob.glob(f"{code_home_folder}{data_folder}monthly_weather.csv")
# weather_array = pd.read_csv(files[0])
# write(title, f"Filename: {files[0]}")
weather_variables = pd.read_csv(glob.glob(f"{code_home_folder}data/processed_arrays/full_dataframe_daily.csv")[0]).drop(["timestamp", 
"building_id", "meter", "meter_reading"], axis=1).columns
# print(weather_variables)
# print("Done.")



# print("\nBUILDING TRAINING DATA")
# write(title, "\nBUILDING TRAINING DATA\n")
# print("Reading dataset...")
# files = glob.glob(f"{code_home_folder}{data_folder}monthly_energy.csv")
# building_array = pd.read_csv(files[0])
# write(title, f"Filename: {files[0]}")
# print("Done.")

# print("\nMODEL TRAIN")
# write(title, "\nMODEL TRAIN\n")
# X = weather_array.to_numpy()
# y = building_array.to_numpy()

# print("Splitting into train and test sets...")
# test_size = 0.33
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
# write(title, f"Training size: {1-test_size}")
# write(title, f"Test size: {test_size}")

# write(title, f"Training array dimensions: {X_train.shape} {y_train.shape}")
# write(title, f"Test array dimensions: {X_test.shape} {y_test.shape}")


# print("Applying normalisation...")
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# y_train = scaler.fit_transform(y_train)
# X_test = scaler.transform(X_test)
# y_test = scaler.transform(y_test)


# write(title, "\nNormalised the training data and applied the same to the test set.")

print("Fitting linear regression model...")
model = LinearRegression(fit_intercept = False).fit(X_train,y_train)
print(f"R\u00b2: {model.score(X_train,y_train)}")
write(title, "\nLinear regression model fit:")
write(title, f"R\u00b2: {model.score(X_train,y_train)}")

y_train_predicted = model.predict(X_train)
print(f"Mean squared error: {mean_squared_error(y_train, y_train_predicted)}")
write(title, f"Mean squared error: {mean_squared_error(y_train, y_train_predicted)}")

print("\nMODEL TEST")
write(title, "\nLinear regression model test:")
print("Predicting the building results...")

y_predicted = model.predict(X_test)

print(f"Mean squared error: {mean_squared_error(y_test, y_predicted)}")
write(title, f"Mean squared error: {mean_squared_error(y_test, y_predicted)}")
print("Model coefficients:")
write(title, "\nModel coefficients:")
coefficients = model.coef_.reshape(-1,1).tolist()
for i in range(len(weather_variables)):
    print(f"{weather_variables[i]} : {coefficients[i]}")
    write(title, f"{weather_variables[i]} : {coefficients[i]}")

print("Model intercept:")
print(model.intercept_)
write(title, f"\nModel intercept: \n{model.intercept_}")
