''' 
MLBP.py is a Multi-layered Feed-forward Neural Network designed to approximate the
Rosenbrock Equation

@author Bryan Downs
'''

# imports
import numpy as np

# figure out how to add parameters for input layer size, output layer size, and hidden layer size
class NeuralNet(object):
    def __init__(self, inputNodes, outputNodes, hiddenNodes, hiddenDepth, learningRate, momentum=False):
        self.inputLayerNodes = inputNodes                     
        self.outputLayerNodes = outputNodes          
        self.hiddenLayerNodes = hiddenNodes
        self.hiddenLayerNumber = hiddenDepth
        self.learningRate = learningRate
        self.momentum = momentum

        # Weight Matrices
        if hiddenDepth == 0:
            self.W1 = np.random.randn(self.inputLayerNodes, self.outputLayerNodes)
        elif hiddenDepth == 1:                                                                      # convert here and after to a list of numpy matrices for dynamic size
            self.W1 = np.random.randn(self.inputLayerNodes, self.hiddenLayerNodes)
            self.WOut = np.random.randn(self.hiddenLayerNodes, self.outputLayerNodes)
        elif hiddenDepth == 2:
            self.W1 = np.random.randn(self.inputLayerNodes, self.hiddenLayerNodes)
            self.W2 = np.random.randn(self.inputLayerNodes, self.outputLayerNodes)
            self.WOut = np.random.randn(self.hiddenLayerNodes, self.outputLayerNodes)

        # additional weight matrices for deeper network

    '''
    

    '''

    # def forward "feeds" the input matrix X through the network
    def forward(self, X):
        # first layer is a dot product multiplication of inputs with the first weight vector
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoidActivation(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.linearActivation(self.z3)
        return yHat

    # sigmoidActivation returns the hyperbolic tangent of input x
    def sigmoidActivation(x):
        return np.tanh(x)

    # sigmoidActivationPrime returns the derivative of sigmoidActivation of input x
    def sigmoidActivationPrime(x):
        return 4/((np.exp(x)-np.exp(-x))**2)

    # the output layer uses a single, linear activation function
    # linearAcivation returns the linear activation of input x
    def linearActivation(x):
        return np.sum(x)

    # linearActivationPrime returns the derivative of linearActivation evaluated for x
    def linearActivationPrime(x):
        return x

    def calculateError(self, X, y):
        # Propogate forward to calculate initial error
        self.yHat = self.forward(X)

        # backpropagate starting at the last layer to update weights ...
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidActivationPrime(self.z3))                      # this might need to be the linear activation function derivative... you not knowing is a problem
        djdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidActivationPrime(self.z2)
        djdW1 = np.dot(X.T, delta2)

        return djdW1, djdW2
    
    '''
    '''




