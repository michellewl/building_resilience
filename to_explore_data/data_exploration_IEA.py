import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import glob



folder = "C:/Users/Michelle/OneDrive/Documents/Uni/MRes/Guided_Team_Challenge/data/IEA"
files = glob.glob(f"{folder}/*.csv")

datasets=[]
datasets.append(pd.read_csv(files[0], header=2, index_col=0))
datasets.append(pd.read_csv(files[1], header=3, index_col=0, usecols=[0,1,2,3,4,5,6,7]))
datasets.append(pd.read_csv(files[2], header=2, index_col=0))
datasets.append(pd.read_csv(files[3], header=3, index_col=0, usecols=[0,1,2,3,4,5,6,7,8,9]))

count=0
for dataset in datasets:
    #print(dataset.head())
    figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    for variable in dataset.columns:
        plt.plot(dataset.index, dataset[variable], label=variable)
    title = files[count].replace(f"{folder}\\", "").replace(".csv", "").replace("-"," ").replace("_", " ")
    plt.title(title)
    plt.legend()
    #plt.show()
    plt.savefig(f"{folder}/{title}.png")
    count+=1


