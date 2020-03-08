#libraries

import glob
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import sklearn.gaussian_process as gp 
from sklearn.preprocessing import StandardScaler
from functions.functions import write, current_time
import pickle
import matplotlib.pyplot as plt  
import matplotlib 
import scipy.io  

#code

folder = "/space/mwlw3/GTC_data_exploration/data_train_test/"
#folder = "C:\\Users\\Michelle\\PycharmProjects\\GTC\\data_train_test\\"

now = current_time()
title = f"GP_log_{now}"

data_type = "buildings"

write(title, f"{current_time()}\nWEATHER TRAINING DATA\n")
write(title, f"All {data_type}. Does not include meta data")

vars = "_no_meta"

write(title, "\nBUILDING TRAINING DATA")
write(title, f"All {data_type}.")

print("Importing data...")
X_train = np.genfromtxt(glob.glob(f"{folder}X_train{vars}.csv")[0], delimiter=",")
y_train = np.genfromtxt(glob.glob(f"{folder}y_train{vars}.csv")[0], delimiter=",")
X_test = np.genfromtxt(glob.glob(f"{folder}X_test{vars}.csv")[0], delimiter=",")
y_test = np.genfromtxt(glob.glob(f"{folder}y_test{vars}.csv")[0], delimiter=",")

write(title, f"Training array dimensions: {X_train.shape} {y_train.shape}")
write(title, f"Test array dimensions: {X_test.shape} {y_test.shape}")


print("Fitting Gaussian process model...")

# set up covariance function
nu = 1
sigma = 1
lambbda = np.exp(-1)

kernel_1 = nu**2 * gp.kernels.RBF(length_scale=lambbda) + gp.kernels.WhiteKernel(noise_level=sigma)

# set up GP model
model_GP = gp.GaussianProcessRegressor(kernel=kernel_1)




write(title, f"{current_time()}\nKernel: {model_GP.kernel}")

model_GP.fit(X_train,y_train)
write(title, f"{current_time()}\nOptimised kernel: {model_GP.kernel}")
write(title, f"log marginal likelihood = {str(round(model_GP.log_marginal_likelihood(),4))}")


filename = f"GP_model{vars}.sav"
pickle.dump(model, open(filename, "wb"))

print("Saved model.")



# predict the mean function with 95% confidence error bars
mean_fn_plotx = np.linspace(-3, 3, 500)
mu, sigma_2 = model_GP.predict(mean_fn_plotx.reshape(-1,1), return_std=True)

fig, ax = plt.subplots(figsize=(10,6))

ax.fill_between(mean_fn_plotx, mu.squeeze() -2*sigma_2, mu.squeeze() +2*sigma_2, alpha=0.2)
ax.plot(mean_fn_plotx, mu.squeeze())
ax.scatter(x, y, color='black')
ax.set_title(model_GP.kernel_)
ax.annotate("log marginal likelihood = "+str(round(model_GP.log_marginal_likelihood(),4)), xy=(x.min(),y.max()*1.05), fontsize=12)
plt.show()