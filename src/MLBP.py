''' 
MLBP.py is a Multi-layered Feed-forward Neural Network designed to approximate the
Rosenbrock Equation

'''

# imports
import numpy as np

# figure out how to add parameters for input layer size, output layer size, and hidden layer size
class Mlbp(object):
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
        
        # add error recording for learning observation
        self.errorList = []

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
        for i in range(1, self.hiddenLayerNumber):
            self.weightMatrixList.append(np.random.randn(self.hiddenLayerNodes, self.hiddenLayerNodes))
        # set dimensions for the output weight matrix with number of output nodes in mind
        self.weightMatrixList.append(np.random.rand(self.hiddenLayerNodes, self.outputLayerNodes))

    '''
    Function Definitions

    '''

    def makeNetwork(self, inputNodes, outputNodes, hiddenNodes, hiddenDepth, learningRate):
        network = MLBP(inputNodes, outputNodes, hiddenNodes, hiddenDepth, learningRate)
        return network
        
    # def forwardPass "feeds" the input matrix X through the network
    def forwardPass(self, X):
        # different cases depending on how many layers are used
        # regardless z0 can be computed by the dot product of the input vector and the first weight matrix
        self.wdotList.append(np.dot(X, self.W0))
        if self.hiddenLayerNumber == 0:
            yHat = self.linearActivation(self.wdotList[0])
        elif self.hiddenLayerNumber == 1:
            self.z2 = np.dot(X, self.weightMatrixList[0])
            self.a2 = self.sigmoidActivation(self.z2)
            self.z3 = np.dot(self.a2, self.weightMatrixList[1])
            yHat = self.linearActivation(self.z3)
        else:
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
        return 1 - np.square((np.tanh(x)))
        
        # return 4/((np.exp(x)-np.exp(-x))**2)

    def linearActivation(self, X):
        return np.sum(X)

    def computeCost(self, X, y):
        self.yHat = self.forwardPass(X)
        cost = 0.5*sum((y-self.yHat)**2)
        return cost

    def gradientDescentUpdate(self, X, y):
        derivativeList = []
        deltaList = []
        # Propogate forwardPass to calculate initial error
        self.yHat = self.forwardPass(X)

        if self.hiddenLayerNumber == 0:
            # only take the derfivative of the linear activation function and adjust
            pass
        elif self.hiddenLayerNumber == 1:
            delta3 = np.multiply(-(y-self.yHat), self.z3)
            derivativeList.insert(0, np.dot(self.a2.T, delta3))

            delta2 = np.dot(delta3, self.weightMatrixList[1].T)*self.sigmoidActivationDerivative(self.z2.T)
            # print(self.sigmoidActivationDerivative(self.z2))
            derivativeList.insert(0, np.dot(X.T, delta2))
        else:
            # start the delta loop, and add each derivative to the beginning of the derivativeList such that it matches the weightMatrixList
            deltaList.insert(0,np.multiply(-(y-self.yHat), 1))
            derivativeList.insert(0, np.dot(self.a2.T, delta3))

            deltaN = np.multiply(-(y-self.yHat), self.linearActivationDerivative(self.wdotList[0]))
            derivativeList.insert(0, np.dot(self.activationList[0]), deltaN)                               
            # backpropagate starting at the last layer to update weights ...
            delta4 = np.multiply(-(y-self.yHat), self.sigmoidActivationDerivative(self.z4))                     # this might need to be the linear activation function derivative... you not knowing is a problem
            djdW3 = np.dot(self.a2.T, delta4)

            delta3 = np.dot(delta4, self.W3.T)*self.sigmoidActivationDerivative(self.z3)                        # this part just chains down
            djdW1 = np.dot(X.T, delta3)

            delta2 = np.dot(delta3, self.W2.T)*self.sigmoidActivationDerivative(self.z2)                        # this part just chains down
            djdW1 = np.dot(X.T, delta2)
        
        for i in range(0, len(derivativeList)):
            derivativeList[i] = np.multiply(derivativeList[i], self.learningRate)
            self.weightMatrixList[i] = np.subtract(self.weightMatrixList[i], derivativeList[i])
            # print(np.multiply(derivativeList[i], self.learningRate)
            # print(np.add(derivativeList[i], self.weightMatrixList[i]))
    
    '''
    Finally, train the network..

    '''
    def train(self, X, y):
        timeStep = 0
        
        for i in range(1000000):
            # call gradient descent
            self.gradientDescentUpdate(X, y)
            # add error and timestep to dictionary
            self.errorList.append((timeStep, abs(y-self.yHat)))
            # repeat 5x to test...
            timeStep += 1
        print(self.errorList)



    def test(self, X, y):
        pass

