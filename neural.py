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

def sigmoid_derivative(x):
    return x - (1.0 - x)

def backpropagate(network, expected_vals):
    for i in range(len(network)-1, -1, -1):
        current_layer = network[i]
        errors = []
        if i != len(network) - 1:
            for j in range(len(current_layer)):
                error = 0.
                for node in network[i + 1]:
                    error += (node.weights[j] * node.delta)
                errors.append(error)
        else:
            for j in range(len(current_layer)):
                node = current_layer[j]
                errors.append(node.collector - expected_vals[j])

        for j in range(len(current_layer)):
            node = current_layer[j]
            node.delta = errors[j] * sigmoid_derivative(node.collector)

def update_weights(network, inputs, lr):
    for i in range(1, len(network)):
        if i != 0:
            inputs = [node.collector for node in network[i-1]]
        for node in network[i]:
            for j in range(len(inputs)):
                node.weights[j] -= lr * node.delta * inputs[j]
            node.weights[-1] -= lr * node.delta
                    


# class to store node attributes
class Node:
    def __init__(self, connections):
        self.connections = connections
        self.collector = 0.0
        self.weights = [random() for x in
                        range(len(connections))] if connections else []
        self.delta = 0.0

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
for _ in range(100):
    error = 0
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
        error += (network[-1][0].collector - expected_val)**2
        backpropagate(network, [expected_val])
        update_weights(network, row, .1)
    print(f"the error is {error}")

# dump that ish
with open('network.pkl', 'wb') as f:
    pickle.dump(network, f)
