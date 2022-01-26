import math
import matplotlib.pyplot as plt
import numpy as np


def relu(x):
    return max(0, x)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def calc_neuron(inputs, weights, bias):
    output = 0
    for i in range(len(weights)):
        output += inputs[i]*weights[i]
    output += bias
    return sigmoid(output) # apply the activator function on the output


def calc_network(neural_network, inputs):

    nn = neural_network

    for layer in range(len(nn)):

        for neuron in range(len(nn[layer])):
            
            if layer > 0:

                prev_layer_vals = []
                for prevneuron in range(len(nn[layer-1])):
                    prev_layer_vals.append(nn[layer-1][prevneuron]["value"])

            nn[layer][neuron]["value"] = calc_neuron(prev_layer_vals if layer>0 else inputs, nn[layer][neuron]["w"], nn[layer][neuron]["b"])

    return nn[-1][0]["value"]


class RandomNeuralNet(object):
    def __init__(self, shape, inputct):
        self.shape = shape
        self.inputct = inputct
        self.N = []
        
        for layer in range(len(self.shape)):
            self.N.append([])
            for neuron in range(self.shape[layer]):
                self.N[layer].append({"w":[], "b":0, "value":None})
                if layer == 0:
                    for weight in range(self.inputct): # if this is the first layer, create a weight for each input
                        self.N[layer][neuron]["w"].append(np.random.normal())
                else:
                    for weight in range(self.shape[layer-1]): # create a weight in the neuron for each neuron in the previous layer
                        self.N[layer][neuron]["w"].append(np.random.normal())

    def spew_contents(self):
        for layer in self.N.values():
            print(layer)

        for layer in self.N.values():
            for neuron in layer.values():
                print(neuron)

nn = RandomNeuralNet([15, 15, 15, 1], 1)
fig = plt.figure()

x=[]
y=[]

size = 20

for i in range(size):

    x.append(i-(size/2))
    y.append(calc_network(nn.N, [i-(size/2)]))

img = plt.plot(x, y)
plt.show()