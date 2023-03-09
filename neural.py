filename = "test.txt"
file_lines = [int(char) for line in open(filename)
              for char in line.strip().split(",")]
network = []
prev_layer = None

for num in file_lines[::-1]:
    layer = []
    for i in range(num):
        layer.insert(0, {"collector": 0.0,
                         "connections": [] if prev_layer == None else prev_layer})
        
    prev_layer = layer
    network.append(layer)

network = network[::-1]
