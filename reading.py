filename = "test.txt"
file_lines = [line.strip().split(",") for line in open(filename)]
file_lines = [int(x) for x in file_lines[0]]
network = []
last_layer = None

for num in file_lines[::-1]:
    layer = []
    for i in range(num):
        layer.insert(0, {"collector": 0.0,
                      "connections": [] if last_layer == None else last_layer})
        
    last_layer = layer
    network.append(layer)

network = network[::-1]
