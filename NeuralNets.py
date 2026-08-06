from math import e
from random import uniform, choice, random

# Neural Network Module v1.2
def ActivationFunction(x, FunctionName="identity"):
    """Multiple activation functions in one!"""
    match FunctionName:
        case "identity":
            return(x)
        case "relu":
            if x >= 0:
                return(x)
            else:
                return(0)
        case "sigmoid":
            return(1 / (1 + e ** (n * -1)))
        case "softplus":
            return(math.log(1 + e ** x))
        case "tanh":
            return((e ** n - e ** (n * -1)) / (e ** n + e ** (n * -1)))
        case "ntanh":
            return((ActivationFunction(x, "tanh") + 1) / 2)

class NeuralNetwork:
    """One Neural Network, 3 hidden layers."""
    def __init__(self, matrix, genePool=None):
        """ Initiates ONE neural network."""
        # Setting up tables
        self.NetworkMatrix = matrix
        self.NodesTable = []
        self.WeightsTable = []
        self.BiasTable = []
        holdinglist = []
        
        # Sets up nodes and biases.
        for i in range(len(matrix)):
            self.NodesTable.append([])
            if i > 0:
                self.BiasTable.append([])
                self.WeightsTable.append([])
            for i2 in range(matrix[i]):
                self.NodesTable[i].append(0)
                if i > 0:
                    self.BiasTable[i-1].append(uniform(-1, 1))
                    for i3 in range(matrix[i-1]):
                        self.WeightsTable[i-1].append(uniform(-1,1))
        
    def GetMatrix(self):
        """Returns the values of the weights, biases, and nodes of a neural network."""
        print("Matrix:", self.NetworkMatrix)
        for i in range(len(self.NetworkMatrix)):
            print("Layer:", i)
            print(self.NodesTable[i])
            if i >= 1:
                print("Weights:", self.WeightsTable[i-1])
                print("Biases:", self.BiasTable[i-1])
        return(self.NodesTable, self.WeightsTable, self.BiasTable)
        
    def ForwardPass(self, inputs, acts):
        """Does a forward pass on a neural network."""
        matrix = self.NetworkMatrix
        weights = self.WeightsTable
        biases = self.BiasTable
        nodes = self.NodesTable.copy()
        storagetable = []
        s = 0
        print("Inputs:")
        for i in range(matrix[0]):
            nodes[0][i] = inputs[i]
            print(nodes[0])
        for i in range(1, len(matrix)):
            for i2 in range(matrix[i]):
                storagetable.clear()
                for i3 in range(matrix[i-1]):
                    storagetable.append(nodes[i-1][i3] * weights[i2 - 1][(i2 * matrix[i-1] + i3)])
                s = sum(storagetable)
                try:
                    nodes[i][i2] = ActivationFunction(s, acts[i])
                except IndexError:
                    nodes[i][i2] = ActivationFunction(s)
        print("Outputs:")
        print(nodes[-1])
        self.NodesTable = nodes
