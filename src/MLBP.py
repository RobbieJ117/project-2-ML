''' 
MLBP.py is a Multi-layered Feed-forward Neural Network designed to approximate the
Rosenbrock Equation

'''

# imports
import numpy as np

# figure out how to add parameters for input layer size, output layer size, and hidden layer size
class MLBP(object):
    def __init__(self, inputNodes, outputNodes, hiddenNodes, hiddenDepth, learningRate):
        # parameter instance variables
        self.inputLayerNodes = inputNodes                     
        self.outputLayerNodes = outputNodes          
        self.hiddenLayerNodes = hiddenNodes
        self.hiddenLayerNumber = hiddenDepth
        self.learningRate = learningRate
        
        # dynamic instance variables
        # store the numpy matrices in lists for dynamic execution of feedfoward and backpropagation
        self.wdotList = []
        self.activationList = []
        self.derivativeList = []

        # finally create the weight matrices...
        # W1 is input nodes x output nodes if no hidden layers are present
        if hiddenDepth == 0:
            self.W0 = np.random.randn(self.inputLayerNodes, self.outputLayerNodes)
        # otherwise W1 is input nodes x hidden nodes in dimension
        else:                                                                      
            self.W0 = np.random.randn(self.inputLayerNodes, self.hiddenLayerNodes)

        # create empty list to hold all the weight matrices
        # the first matrix dimension is conditional on number of hidden layers above
        self.weightMatrixList = [self.W0]
        # add hidden layers of hiddenNode x hiddenNode dimensions for each hidden layer greater than 1
        for i in range(self.hiddenLayerNumber):
            self.weightMatrixList.append(np.random.randn(self.hiddenLayerNodes, self.hiddenLayerNodes))
        # set dimensions for the output weight matrix with number of output nodes in mind
        self.weightMatrixList.append(np.random.rand(self.hiddenLayerNodes, self.outputLayerNodes))

    '''
    Function Definitions

    '''
    # def forwardPass "feeds" the input matrix X through the network
    def forwardPass(self, X):
        # different cases depending on how many layers are used
        # regardless z0 can be computed by the dot product of the input vector and the first weight matrix
        self.wdotList.append(np.dot(X, self.W0))
        if self.hiddenLayerNumber == 0:
            yHat = self.linearActivation(self.wdotList[0])
        if self.hiddenLayerNumber > 0:
            # iterate until all weight matrix dot products have been carried out
            for i in range(1, len(self.weightMatrixList)):
                # need to get a z and an a
                # add the first activation function evaluation of I * W 
                self.activationList.append(self.sigmoidActivation(self.wdotList[i-1]))
                # compute the next weight * a dot product
                self.wdotList.append(np.dot(self.activationList[i-1], self.weightMatrixList[i]))
            # the last item in wdotList is the precursor to the output, and needs linear activation function
            yHat = self.linearActivation(self.wdotList(len(self.wdotList) - 1))

        # return the estimation of y
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

    def gradientDescent(self, X, y):
        # Propogate forwardPass to calculate initial error
        self.yHat = self.forwardPass(X)

        # backpropagate starting at the last layer to update weights ...
        delta4 = np.multiply(-(y-self.yHat), self.sigmoidActivationDerivative(self.z4))                     # this might need to be the linear activation function derivative... you not knowing is a problem
        djdW3 = np.dot(self.a2.T, delta4)

        delta3 = np.dot(delta4, self.W3.T)*self.sigmoidActivationDerivative(self.z3)                        # this part just chains down
        djdW1 = np.dot(X.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidActivationDerivative(self.z2)                        # this part just chains down
        djdW1 = np.dot(X.T, delta2)

        return djdW1, djdW2
    
    '''
    Finally, train the network..

    '''
    def train(self, X, y):
        pass
    




