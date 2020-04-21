# This script defines a class used to help compile an evaluation table in the plot_model_evaluation script.

import re
from multilayer_perceptron import SimpleNet_2, SimpleNet_3, SimpleNet_4

class NNetwork():
    def __init__(self, filename):
        self.filename = filename
        self.nodes = list(map(int, re.compile(r"\d+").findall(str(re.compile("_"+r"\d+").findall(filename)))))
        self.hidden_layers = len(self.nodes)

        if self.hidden_layers == 2:
            self.model = SimpleNet_2(49, self.nodes[0], self.nodes[1])
        elif self.hidden_layers ==3:
            self.model = SimpleNet_3(49, self.nodes[0], self.nodes[1], self.nodes[2])
        elif self.hidden_layers ==4:
            self.model = SimpleNet_4(49, self.nodes[0], self.nodes[1], self.nodes[2], self.nodes[3])

        self.n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
