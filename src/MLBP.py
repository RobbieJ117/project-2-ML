''' 
MLBP.py is a Multi-layered Feed-forward Neural Network designed to approximate the
Rosenbrock Equation

'''

# imports
import numpy as np

# figure out how to add parameters for input layer size, output layer size, and hidden layer size
class MLBP(object):
    def __init__(self, inputNodes, outputNodes, hiddenNodes, hiddenDepth, learningRate, momentum=0):
        self.inputLayerNodes = inputNodes                     
        self.outputLayerNodes = outputNodes          
        self.hiddenLayerNodes = hiddenNodes
        self.hiddenLayerNumber = hiddenDepth
        self.learningRate = learningRate
        self.momentum = momentum

        # W1 is input nodes x output nodes if no hidden layers are present
        if hiddenDepth == 0:
            self.W1 = np.random.randn(self.inputLayerNodes, self.outputLayerNodes)
        # otherwise W1 is input nodes x hidden nodes in dimension
        else:                                                                      
            self.W1 = np.random.randn(self.inputLayerNodes, self.hiddenLayerNodes)

        # create empty list to hold all the weight matrices
        # the first matrix dimension is conditional on number of hidden layers above
        self.weightMatrixList = [self.W1]
        # add hidden layers of hiddenNode x hiddenNode dimensions for each hidden layer greater than 1
        for i in range(self.hiddenLayerNumber):
            self.weightMatrixList.append(np.random.randn(self.hiddenLayerNodes, self.hiddenLayerNodes))
        # set dimensions for the output weight matrix with number of output nodes in mind
        self.weightMatrixList.append(np.random.rand(self.hiddenLayerNodes, self.outputLayerNodes))

    '''
    

    '''
    # def forwardPass "feeds" the input matrix X through the network
    def forwardPass(self, X):
        # different cases depending on how many layers are used
        # first case is for 2 hidden layers
        if self.hiddenLayerNumber == 2:
            self.z2 = np.dot(X, self.W1)
            self.a2 = self.sigmoidActivation(self.z2)
            self.z3 = np.dot(self.a2, self.W2)
            self.a3 = self.sigmoidActivation(self.z3)
            self.z4 = np.dot(self.a3, self.W3)
            yHat = self.linearActivation(self.z4)
        # next case is for one hidden layer
        elif self.hiddenLayerNumber == 1:
            self.z2 = np.dot(X, self.W1)
            self.a2 = self.sigmoidActivation(self.z2)
            self.z3 = np.dot(self.a2, self.W2)
            yHat = self.linearActivation(self.z3)
        elif self.hiddenLayerNumber == 0:
            self.z2 = np.dot(X, self.W1)
            yHat = self.linearActivation(self.z2)
        
        return yHat

    # sigmoidActivation returns the hyperbolic tangent of input x
    def sigmoidActivation(self, x):
        return np.tanh(x)

    # sigmoidActivationDerivative returns the derivative of sigmoidActivation of input x
    def sigmoidActivationDerivative(self, x):
        return 4/((np.exp(x)-np.exp(-x))**2)

    # the output layer uses a single, linear activation function
    # linearAcivation returns the linear activation of input x
    def linearActivation(self, x):
        return np.sum(x)

    # linearActivationPrime returns the derivative of linearActivation evaluated for x
    def linearActivationDerivative(self, x):
        return x

    def computeCost(self, X, y):
        self.yHat = self.forwardPass(X)
        cost = 0.5*sum((y-self.yHat)**2)
        return cost

    def computeCostDerivative(self, X, y):
        # Propogate forwardPass to calculate initial error
        self.yHat = self.forwardPass(X)

        # backpropagate starting at the last layer to update weights ...
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidActivationDerivative(self.z3))                      # this might need to be the linear activation function derivative... you not knowing is a problem
        djdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidActivationDerivative(self.z2)
        djdW1 = np.dot(X.T, delta2)

        return djdW1, djdW2
    
    '''
    Finally, train the network..

    '''
    def train(self, X, y):
        pass
    




