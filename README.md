# neural-net-module-v1.1
Python Module for creating Neural Networks.

Documentation:

ActivationFunction(x, FunctionName="identity"): Used for the ForwardPass() function (see below.)
x: The x value of the function.
FunctionName: The type of function that x is an input for. This supports identity, softplus, relu, sigmoid, tanh, and normalized tanh (written as ntanh.)

NeuralNetwork(matrix, genePool=None): Creates a neural network.
matrix: The number of nodes on each layer of the neural network. The number of items in matrix determines the number of items in the neural network.
genePool: The table of weights that will be used for the neural network (chooses randomly.) If this is set to none, randomly generates weights with float values from -1 to 1.

GetMatrix(): Prints and returns the nodes, weights, and biases of the neural network. (Returns a tuple of length 3.)

ForwardPass(inputs, acts): Does a forward pass of the neural network.
inputs: The values for the input nodes. MUST be an iterable with length equal to the amount of input nodes of the network.
acts: The types of activation functions used. Is an iterable with length equal to the layers of the neural network - 1.
