import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import randn
from scipy import array, newaxis

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

nn = RandomNeuralNet([15, 15, 15, 1], 2)

samples = 40
scale = 0.5

x=[]
y=[]
z=[]

for i in range(samples):
    for j in range(samples):
        x.append((i*scale)-(samples*scale/2))
        y.append((j*scale)-(samples*scale/2))
        z.append(calc_network(nn.N, [(i*scale)-(samples*scale/2), (j*scale)-(samples*scale/2)]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0)
fig.colorbar(surf)

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))

fig.tight_layout()

plt.show()