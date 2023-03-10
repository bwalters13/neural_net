struct_filename = "test.txt"
input_filename = "input.txt"
struct_file_lines = [int(char) for line in open(struct_filename)
              for char in line.strip().split(",")]
network = []
prev_layer = None

for num in struct_file_lines:
    layer = []
    for i in range(num):
        layer.append({"collector": 0.0,
                         "connections": [] if prev_layer == None else prev_layer})
        
    prev_layer = layer
    network.append(layer)

input_file_lines = [float(char) for line in open(input_filename)
              for char in line.strip().split(",")]


for i, val in enumerate(input_file_lines):
    network[0][i]['collector'] = val


for i in range(1, len(struct_file_lines)):
    for j in range(len(network[i])):
        for node in network[i][j]['connections']:
            network[i][j]['collector'] = network[i][j]['collector'] + node['collector']

print(network[2][0]['collector'])
    
