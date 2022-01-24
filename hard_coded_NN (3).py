import numpy as np


def calc_neuron(inputs, weights, bias):
    output = 0
    for i in range(len(weights)):
        output += inputs[i]*weights[i]
    output += bias
    return max(0, output) # apply the relu activator function on the output

# def old_calc_network(neural_network, inputs):
#     l1_vals = []
#     l2_vals = []
#     output = 0
#     for neuron in neural_network["l1"].values():
#         l1_vals.append(calc_neuron(inputs, neuron["weights"], neuron["bias"]))
#     for neuron in neural_network["l2"].values():
#         l2_vals.append(calc_neuron(l1_vals, neuron["weights"], neuron["bias"]))
#     return calc_neuron(l2_vals, neural_network["output"]["weights"], neural_network["output"]["bias"])

def calc_network(neural_network, inputs):
    nn = neural_network
    currentlayer = 0
    for layer in nn.values():
        currentneuron = 0
        for neuron in layer.values():
            if currentlayer > 0:
                prev_layer_vals = []
                for neuron in nn[currentlayer-1].values():
                    prev_layer_vals.append(neuron["value"])

            print(nn[nn.keys()[0]])

            nn[currentlayer][currentneuron]["value"] = calc_neuron(prev_layer_vals if currentlayer>0 else inputs, neuron["w"], neuron["b"])
            currentneuron += 1
        currentlayer += 1


class RandomNeuralNet(object):
    def __init__(self, shape, inputct):
        self.shape = shape
        self.inputct = inputct
        self.N = {}
        
        for layer in range(len(self.shape)):
            self.N["l" + str(layer)] = {}
            for neuron in range(self.shape[layer]):
                self.N["l" + str(layer)]["n" + str(neuron)] = {}
                self.N["l" + str(layer)]["n" + str(neuron)]["b"] = 0 # give neurons a bias of zero
                self.N["l" + str(layer)]["n" + str(neuron)]["value"] = None # give neurons an empty value, used for calculating the end result
                self.N["l" + str(layer)]["n" + str(neuron)]["w"] = [] # create an array for the weights
                if layer == 0:
                    for weight in range(self.inputct): # if this is the first layer, create a weight for each input
                        self.N["l" + str(layer)]["n" + str(neuron)]["w"].append(5*(np.random.rand()-0.5))
                else:
                    for weight in range(self.shape[layer-1]): # create a weight in the neuron for each neuron in the previous layer
                        self.N["l" + str(layer)]["n" + str(neuron)]["w"].append(5*(np.random.rand()-0.5))

    def spew_contents(self):
        for layer in self.N.values():
            print(layer)

        for layer in self.N.values():
            for neuron in layer.values():
                print(neuron)
    
testNN = RandomNeuralNet([2, 2, 1], 3)

# testNN.spew_contents()

print(calc_network(testNN.N, [10, 10, 10]))