import math
from math import e
import random
from random import uniform, choice, random

# Neural Network Module v1.1
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

WeightsPool = []

class NeuralNetwork:
    """One Neural Network, 3 hidden layers."""
    def __init__(self, inp, h1, h2, h3, o, genePool=None):
        """ Initiates ONE neural network."""
        # Setting up tables
        self.NetworkMatrix = [inp, h1, h2, h3, o]
        self.NodesTable = [[], [], [], [], []]
        self.WeightsTable = [[], [], [], []]
        self.BiasTable = [[],[],[],[]]
        self.Fitness = 0
        nm = self.NetworkMatrix
        # Putting stuff into the tables
        for i in range(nm[0]):
            self.NodesTable[0].append(0)
            
        for i in range(nm[1]):
            self.NodesTable[1].append(0)
            self.BiasTable[0].append(random())
        for j in range(nm[0] * nm[1]):
            if genePool == None:
               self.WeightsTable[0].append(uniform(-1, 1))
            else:
                self.WeightsTable[0].append(choice(genePool))
                    
        for i in range(nm[2]):
            self.NodesTable[2].append(0)
            self.BiasTable[1].append(random())
        for j in range(nm[1] * nm[2]):
            if genePool == None:
                self.WeightsTable[1].append(uniform(-1,1))
            else:
                self.WeightsTable[1].append(choice(genePool))
                    
        for i in range(nm[3]):
            self.NodesTable[3].append(0)
            self.BiasTable[2].append(random())
        for j in range(nm[2] * nm[3]):
            if genePool == None:
                self.WeightsTable[2].append(uniform(-1,1))
            else:
                self.WeightsTable[2].append(choice(genePool))
                    
        for i in range(nm[4]):
            self.NodesTable[4].append(0)
            self.BiasTable[3].append(random())
        for j in range(nm[3] * nm[4]):
            if genePool == None:
               self.WeightsTable[3].append(uniform(-1,1))
            else:
                self.WeightsTable[3].append(choice(genePool))
    def GetMatrix(self):
        """Returns the values of the weights, biases, and nodes of a neural network."""
        print("Matrix:", self.NetworkMatrix)
        for i in range(5):
            print("Layer:", i)
            print(self.NodesTable[i])
            if i >= 1:
                print("Weights:", self.WeightsTable[i-1])
                print("Biases:", self.BiasTable[i-1])
        return(self.NodesTable, self.WeightsTable, self.BiasTable)
        
    def ForwardPass(self, inputs):
        """A forward pass in a neural network."""
        matrix = self.NetworkMatrix # We don't need a shallow copy of NetworkMatrix because we aren't going to change it.
        nodes = self.NodesTable.copy()
        weights = self.WeightsTable.copy()
        biases = self.BiasTable.copy()
        sums = None
        storagelist = []
        print("Inputs:", inputs)
        for i in range(len(inputs)):
            nodes[0][i] = inputs[i]
        for i in range(matrix[1]):
            currweights = weights.copy()[i * matrix[0]:(i + 1) * matrix[0]]
            for i2 in range(matrix[0]):
                storagelist.append(nodes[0][i2])
            storagelist.append(biases[0][i])
            sums = sum(storagelist)
            nodes[1][i] = ActivationFunction(sums)
        storagelist.clear()
        for i in range(matrix[2]):
            currweights = weights.copy()[i * matrix[1]:(i + 1) * matrix[1]]
            for i2 in range(matrix[1]):
                storagelist.append(nodes[1][i2])
            storagelist.append(biases[1][i])
            sums = sum(storagelist)
            nodes[2][i] = ActivationFunction(sums)
        storagelist.clear()
        for i in range(matrix[3]):
            currweights = weights.copy()[i * matrix[2]:(i + 1) * matrix[2]]
            for i2 in range(matrix[2]):
                storagelist.append(nodes[2][i2])
            storagelist.append(biases[2][i])
            sums = sum(storagelist)
            nodes[3][i] = ActivationFunction(sums)
        storagelist.clear()
        for i in range(matrix[4]):
            currweights = weights.copy()[i * matrix[3]:(i + 1) * matrix[3]]
            for i2 in range(matrix[3]):
                storagelist.append(nodes[3][i2])
            storagelist.append(biases[3][i])
            sums = sum(storagelist)
            nodes[4][i] = ActivationFunction(sums)
        print("Outputs:", nodes[4])

        for i in range(5):
            self.NodesTable[i] = nodes[i]


                



                
