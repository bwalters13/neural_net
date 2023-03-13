struct_filename = "test.txt"
number_input_file = "input.txt"
struct_file_lines = [int(char) for line in open(struct_filename)
              for char in line.strip().split(",")]
network = []
prev_layer = None

class Node:
    def __init__(self, connections):
        self.connections = connections
        self.collector = 0.0
    def set_value(self, value):
        self.collector = value

for num in struct_file_lines:
    layer = []
    for i in range(num):
        connections = [] if prev_layer == None else prev_layer
        layer.append(Node(connections))
        #layer.append({"collector": 0.0,
        #                 "connections": [] if prev_layer == None else prev_layer})
        
    prev_layer = layer
    network.append(layer)


#print(network[1][0].connections)
number_inputs = [float(char) for line in open(number_input_file)
              for char in line.strip().split(",")]


for i, val in enumerate(number_inputs):
    network[0][i].set_value(val)


for i in range(1, len(struct_file_lines)):
    for j in range(len(network[i])):
        for node in network[i][j].connections:
            network[i][j].set_value(network[i][j].collector + node.collector)

print(network[2][0].collector
      )
    
