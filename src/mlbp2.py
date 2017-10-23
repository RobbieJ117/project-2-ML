''' 
mlbp2.py is a Multi-layered Feed-forward Neural Network designed to approximate theRosenbrock Equation

@author Bryan Downs
@sources adapted with reference to Michael Nielsen's book at neuralnetworksanddeeplearning.com, 
        specifically: http://neuralnetworksanddeeplearning.com/chap1.html#implementing_our_network_to_classify_digits
'''

# import packages
import random
import numpy as np

import matplotlib.pyplot as plt
import os.path

# class init
class mlbp2(object):
    def __init__(self, layers, id):
        # derive the number of layers from the length of the input parameter sizes
        self.layerNum = len(layers)
        self.layerSizes = layers
        self.id = id
        self.lossHistory = []
        self.epochs = 100

        # initialize weight and biase matrices
        self.biases = [np.square(2/layers[0]) * np.random.randn(y, 1) for y in layers[1:]]
        
        self.weights = [np.square(2/layers[0]) * np.random.randn(y, x)
                        for x, y in zip(layers[:-1], layers[1:])]
        
        print(self.weights)
        print(self.biases)

    '''
    Activation Function Defs

    ''' 
    # hypTangent returns the hyperbolic tangent of input x
    def hypTangent(self, x):
        return np.tanh(x)

    # hypTangentPrime returns the derivative of tanh of input x
    def hypTangentPrime(self, x):
        htp = 1 - np.square((np.tanh(x)))
        # print(htp)
        return htp

    # linear activation is just the pre-activation matrix
    def linearActivation(self, X):
        return X
    
    # derivative of the pre-activation matrix is an equally dimensioned matrix of ones
    def linearActivationPrime(self, X):
        matrix = np.ones_like(X)
        return matrix

    

    # derivative of the cost function for the nth step
    def costPrime(self, outputActivations, y):
        return  -1*(y - outputActivations)


    '''
    Network Traversal Defs

    '''
    # def forwardPass "feeds" the input matrix X through the network
    # especially useful for single pass testing
    def forwardPass(self, activation):
        # grab the last bias weight pair before ziping list together for iteration
        lBias = self.biases[-1]
        lWeight = self.weights[-1]
        
        # prepare all tanh/sigmoid weights in temp fields by removing the last item in each
        tBiases = self.biases[:-1]
        tWeights = self.weights[:-1]

        # zip together temp bias/weight pairs for the sigmoid function and iterate
        for bias, weight in zip(tBiases, tWeights):
            activation = self.hypTangent(np.dot(weight, activation) + bias)

        # compute the final pass with the linear activation function
        activation = self.linearActivation(np.dot(lWeight, activation) + lBias)
        return activation

    
    def backpropagate(self, x, y):
        # store partial derivatives with respect to bias and weight terms
        dwrt_bias = [np.zeros(b.shape) for b in self.biases]
        dwrt_weight = [np.zeros(w.shape) for w in self.weights]

        # forwardPass
        # init activation list for backprop setup
        activation = x
        activationList = [x]
        # print('--------------------')
        # print('input: {}'.format(activation))
        # print('expected output: {}'.format(y))
        # print('calculated output: {}'.format(self.forwardPass(activation)))
        # print('error: {}'.format(y - self.forwardPass(activation)))
        # print()

        # grab the last bias weight pair before ziping list together for iteration
        lBias = self.biases[-1]
        lWeight = self.weights[-1]
        
        # prepare all tanh/sigmoid weights in temp fields by removing the last item in each
        tBiases = self.biases[:-1]
        tWeights = self.weights[:-1]

        # stores the intermediate evaluation of the weights times previous layer activation,
        # prior to the activation step
        zList = []
        for bias, weight in zip(tBiases, tWeights):
            z = np.dot(weight, activation) + bias
            zList.append(z)
            activation = self.hypTangent(z)
            # print(activation)
            activationList.append(activation)
        
        # compute the final pass with the linear activation function
        z = np.dot(lWeight, activation) + lBias
        zList.append(z)
        activation = self.linearActivation(z)
        activationList.append(activation)

        # backprop starts
        # traversing previous activation and intermediate lists in negative index as in the book
        # a key difference here is the derivative of the linear function, as we're doing function 
        # approximation instead of classification
        delta = self.costPrime(activationList[-1], y)
        dwrt_bias[-1] = delta
        dwrt_weight[-1] = np.dot(delta, activationList[-2].T)

        # loop for dynamic size, again with a decreasing index to be reverse of add order
        for l in range(2, self.layerNum):
            z = zList[-l]
            htp = self.hypTangentPrime(z)
            delta = np.dot(self.weights[-l+1].T, delta) * htp
            dwrt_bias[-l] = delta
            dwrt_weight[-l] = np.dot(delta, activationList[-l-1].T)

        # return the derivative with respect to bias and weight matrices
        return dwrt_bias, dwrt_weight

    '''
    Finally, train the network using mini-batch training.
    Parameters are the batch and a learning rate (batch, learningRate), respectively.

    '''

    def stochGradientDescent(self, trainingData, epochs, batchSize, learningRate, testData=None):
        n = len(trainingData)
        self.epochs = epochs
        for j in range(epochs):
            random.shuffle(trainingData)
            batches = [
                trainingData[k:k+batchSize]
                for k in range(0, n, batchSize)
            ]
            for batch in batches:
                # anneal this.
                # learningRate = learningRate / (j+1)
                self.updateBatch(batch, learningRate)
            if testData is not None:
                self.test(testData)
            else:
                print("Epoch {} complete".format(j))    
        self.print_results()

    def updateBatch(self, batch, learningRate):
        dwrt_bias = [np.zeros(b.shape) for b in self.biases]
        dwrt_weight = [np.zeros(w.shape) for w in self.weights]

        for x in batch:
            iVector = np.zeros(((len(x) - 1),1))
            for i in range(len(x) - 1):
                iVector[i] = x[i]
            delta_dwrt_bias, delta_dwrt_weight = self.backpropagate(iVector, x[-1])
            dwrt_bias = [db+ddb for db, ddb in zip(dwrt_bias, delta_dwrt_bias)]
            dwrt_weight = [dw+ddw for dw, ddw in zip(dwrt_weight, delta_dwrt_weight)]
        
        # print (delta_dwrt_weight, delta_dwrt_bias)
        # update weight matrices after the batch is complete
        # adjustment amount is normalized by the batch size
        self.weights = [w - ((learningRate/len(batch))*dw) 
                            for w, dw in zip(self.weights, dwrt_weight)]
        self.biases = [b - ((learningRate/len(batch))*db)
                            for b, db in zip(self.biases, dwrt_bias)]


    '''
    Test method to find accuracy

    '''

    def test(self, testData):
        test_results = 0
        yValList = 0
        for x in testData:
            xval = np.zeros((len(x) - 1, 1))
            for i in range(len(x) - 1):
                xval[i] = x[i]
            yval = float(x[len(x) - 1])
            # print(xval)
            # print(self.forwardPass(xval))
            test_results = test_results + np.asscalar((self.forwardPass(xval) - yval)**2)
            yValList = yValList +  yval
        # print(yValList/len(testData))
        MSE = test_results/(len(testData))
        self.lossHistory.append(MSE)
        return MSE



    def print_results(self):
        file_name = "{}.png".format(self.id)
        fig = plt.figure()
        plt.plot(np.arange(0, len(self.lossHistory)), self.lossHistory)
        fig.suptitle("Training Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        fig.savefig(file_name)
        plt.close(fig)
        self.lossHistory = []
        print(self.weights)
        print(self.biases)