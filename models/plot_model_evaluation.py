import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from functions.functions import get_models_by_hidden_layers

windows_os = True

if windows_os:
    code_home_folder = "C:\\Users\\Michelle\\OneDrive - University of Cambridge\\MRes\\Guided_Team_Challenge\\building_resilience\\"
    folder = f"{code_home_folder}logs\\assessment\\daily_data\\"
    models_folder = f"{code_home_folder}models\\saved\\"
else:
    code_home_folder = "/home/mwlw3/Documents/Guided_team_challenge/building_resilience/"
    folder = f"{code_home_folder}logs/assessment/daily_data/"
    models_folder = f"{code_home_folder}models/saved/"

filepath = f"{folder}Model_evaluation.csv"

eval_table = pd.read_csv(filepath)

models_list = glob(f"{models_folder}*.tar")

two_hl_list = get_models_by_hidden_layers(models_list, 2)
three_hl_list = get_models_by_hidden_layers(models_list, 3)
four_hl_list = get_models_by_hidden_layers(models_list, 4)



#
# fig, ax = plt.subplots()
# ax.scatter(eval_table.loc[eval_table.model_type == "multilayer_perceptron"].index,
#            eval_table.loc[eval_table.model_type == "multilayer_perceptron"].smape, label="neural network")
# ax.scatter(eval_table.loc[eval_table.model_type == "linear_regression"].index,
#            eval_table.loc[eval_table.model_type == "linear_regression"].smape, label="linear regression")
# plt.ylim(0, 100)
# plt.ylabel("% error (SMAPE metric)")
# plt.xlabel("Model complexity / arb. units")
# plt.legend()
# plt.show()
