import numpy as np
import glob

folder = "train_test_arrays"

for file in glob.glob(f"{folder}\\*.csv"):
    print(file)
    array = np.genfromtxt(file, delimiter=",")
    np.save(file.replace(".csv", ".npy"), array)
    print("Saved as npy file.")