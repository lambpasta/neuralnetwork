default_weight = 1
default_bias = 1


neural_network = {
    "l1": {"n1":{"weights":{"w1": default_weight, "w2": default_weight, "w3": default_weight}, "bias":default_bias}, "n2":{"weights":{"w1": default_weight, "w2": default_weight, "w3": default_weight}, "bias":default_bias}},
    "l2": {"n1":{"weights":{"w1": default_weight, "w2": default_weight}, "bias":default_bias}, "n2":{"weights":{"w1": default_weight, "w2": default_weight}, "bias":default_bias}},
    "l3": {"n1":{"weights":{"w1": default_weight, "w2": default_weight}, "bias":default_bias}}
}

print(neural_network)