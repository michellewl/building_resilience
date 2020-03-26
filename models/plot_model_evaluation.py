import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from functions.functions import get_models_by_hidden_layers
import torch
import re
from neural_networks.multilayer_perceptron import SimpleNet_2, SimpleNet_3, SimpleNet_4

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


class NNetwork():
    def __init__(self, filename):
        self.filename = filename
        self.nodes = list(map(int, re.compile(r'\d+').findall(filename)))
        self.hidden_layers = len(self.nodes)

        if self.hidden_layers == 2:
            self.model = SimpleNet_2(49, self.nodes[0], self.nodes[1])
        elif self.hidden_layers ==3:
            self.model = SimpleNet_3(49, self.nodes[0], self.nodes[1], self.nodes[2])
        elif self.hidden_layers ==4:
            self.model = SimpleNet_4(49, self.nodes[0], self.nodes[1], self.nodes[2], self.nodes[3])

        self.n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

# two_hl = NNetwork(two_hl_list[0])
# print(two_hl.hidden_layers, two_hl.nodes)
# print(two_hl.nodes[0])
# print(two_hl.n_parameters)
#


for model_name in models_list:
    model = NNetwork(model_name)
    print(f"\nNumber of hidden layers: {model.hidden_layers}"
          f"\nNumber of nodes per hidden layer: {model.nodes}"
          f"\nNumber of parameters: {model.n_parameters}")


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
