# GenFunc-Approximator
 Neural network that can approximate both single variable and
 multivariable continuous functions. Neural Network has an input layer,
 two hidden layers and an output layer, with an arbitrary amount of nodes
 (dictated by the amount of weights in the checkpoints file). Layer 1
 nodes use the function (X^W1 + B1, X is the sum of inputs, W1 is the weight 
 of a Layer 1 node, and B1 is a bias of a Layer 1 node) and Layer 2 nodes
 use the function (X*W2 + B1, X is the sum of Layer 1 nodes, W2 is the 
 weight of a Layer 2 node, and B2 is a bias of a Layer 2 node).
