def relu(x): # the relu function is one of several activation functions that can be used as activators
    return max(0, x)

def calc_neuron(inputs, weights, bias):
    output = 0 # declare a variable to accumulate the output into
    for i in range(len(weights)):
        output += inputs[i]*weights[i] # multiply all the inputs by weights and add them up
    output += bias # add the bias
    return relu(output) # apply the relu activator function on the output

def forward_propagation(neural_network, inputs):
    l1_vals = [] # outputs of the layer one neurons
    l2_vals = [] # outputs of the layer two neurons
    output = 0
    for neuron in neural_network["l1"].values(): # for each layer 1 neuron
        l1_vals.append(calc_neuron(inputs, neuron["weights"], neuron["bias"]))
        # call the calc_neuron func. on each with the inputs as inputs
    for neuron in neural_network["l2"].values(): # for each layer 1 neuron
        l2_vals.append(calc_neuron(l1_vals, neuron["weights"], neuron["bias"]))
        # call the calc_neuron func. on each with previous outputs as inputs
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
print(forward_propagation(neural_network, inputs))