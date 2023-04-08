import pickle
from random import random
import math
import numpy as np
import pandas as pd


# files to get structure and inputs
struct_filename = "test.txt"
number_input_file = "input.txt"
struct_file_lines = [int(char) for line in open(struct_filename)
                     for char in line.strip().split(",")]
network = []
prev_layer = None


def sigmoid(x):
    return 1/(1+math.exp(-x))


# class to store node attributes
class Node:
    def __init__(self, connections):
        self.connections = connections
        self.collector = 0.0
        self.weights = [random() for x in
                        range(len(connections))] if connections else []

    def set_value(self, value):
        self.collector = value


# create that structure
for num in struct_file_lines:
    layer = []
    for i in range(num):
        connections = [] if prev_layer is None else prev_layer
        layer.append(Node(connections))
    prev_layer = layer
    network.append(layer)

number_inputs = [float(char) for line in open(number_input_file)
                 for char in line.strip().split(",")]

data = pd.DataFrame(np.array(number_inputs).reshape(-1, len(network[0])+1),
                    columns=[*range(4), 'answer'])

expected_answers = data.answer.values
number_inputs = data.drop('answer', axis=1).values



# set inputs
for row, expected_val in zip(number_inputs, expected_answers):
    for i, val in enumerate(row):
        network[0][i].set_value(val)

    # adding!!!
    for i in range(1, len(struct_file_lines)):
        for j in range(len(network[i])):
            for node, weight in zip(network[i][j].connections,
                                    network[i][j].weights):
                network[i][j].set_value(network[i][j].collector
                                        + weight * node.collector)
            network[i][j].set_value(sigmoid(network[i][j].collector))
    # we did it??
    for i in range(len(network[-1])):
        print(network[-1][i].collector)
    error = expected_val - network[-1][i].collector
    print(f"The error is {error}")

# dump that ish
with open('network.pkl', 'wb') as f:
    pickle.dump(network, f)
