import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

windows_os = True

if windows_os:
    code_home_folder = "C:\\Users\\Michelle\\OneDrive - University of Cambridge\\MRes\\Guided_Team_Challenge\\building_resilience\\"
    folder = f"{code_home_folder}logs\\assessment\\daily_data\\"
else:
    code_home_folder = "/home/mwlw3/Documents/Guided_team_challenge/building_resilience/"
    folder = f"{code_home_folder}logs\\assessment\\daily_data\\"

filepath = f"{folder}Model_evaluation.csv"

eval_table = pd.read_csv(filepath)


