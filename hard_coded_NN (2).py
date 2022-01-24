import numpy as np


def calc_neuron(inputs, weights, bias):
    output = 0
    for i in range(len(weights)):
        output += inputs[i]*weights[i]
    output += bias
    return max(0, output) # apply the relu activator function on the output

def calc_network(neural_network, inputs):
    l1_vals = []
    l2_vals = []
    output = 0
    for neuron in neural_network["l1"].values():
        l1_vals.append(calc_neuron(inputs, neuron["weights"], neuron["bias"]))
    for neuron in neural_network["l2"].values():
        l2_vals.append(calc_neuron(l1_vals, neuron["weights"], neuron["bias"]))
    return calc_neuron(l2_vals, neural_network["output"]["weights"], neural_network["output"]["bias"])


    
default_weight = 1
default_bias = 1

neural_network = {
    "l1": {
        "n1":{
            "weights":[
                default_weight,
                default_weight,
                default_weight],
            "bias":default_bias},
        "n2":{
            "weights":[
                default_weight,
                default_weight,
                default_weight],
            "bias":default_bias}},
    "l2": {
        "n1":{
            "weights":[
                default_weight,
                default_weight
            ],
            "bias":default_bias},
        "n2":{
            "weights":[
                default_weight,
                default_weight],
            "bias":default_bias}},
    "output": {
        "weights":[
            default_weight,
            default_weight],
        "bias":default_bias}
}

inputs = [1, 1, 1]
print(calc_network(neural_network, inputs))