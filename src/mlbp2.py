''' 
mlbp2.py is a Multi-layered Feed-forward Neural Network designed to approximate theRosenbrock Equation

@author Bryan Downs
@sources adapted with reference to Michael Nielsen's book at neuralnetworksanddeeplearning.com, 
        specifically: http://neuralnetworksanddeeplearning.com/chap1.html#implementing_our_network_to_classify_digits
'''

# import packages
import numpy as np

# class init
class mlbp2(object):
    def __init__(self, layers):
        # derive the number of layers from the length of the input parameter sizes
        self.layerNum = len(layers)
        self.layerSizes = layers

        # initialize weight and biase matrices
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layers[:-1], layers[1:])]

    '''
    Activation Function Defs

    ''' 
    # hypTangent returns the hyperbolic tangent of input x
    def hypTangent(self, x):
        return np.tanh(x)

    # hypTangentPrime returns the derivative of tanh of input x
    def hypTangentPrime(self, x):
        return 1 - np.square((np.tanh(x)))
        # return 4/((np.exp(x)-np.exp(-x))**2)

    # linear activation sums the vector X into a scalar result
    def linearActivation(self, X):
        return X

    def linearActivationPrime(self, X):
        matrix = np.ones_like(X)
        return matrix


    '''
    Network Traversal Defs

    '''
    # def forwardPass "feeds" the input matrix X through the network
    def forwardPass(self, activation):
        # grab the last bias weight pair before ziping list together for iteration
        lBias = self.biases[-1]
        lWeight = self.weights[-1]
        
        # prepare all tanh/sigmoid weights in temp fields by removing the last item in each
        tBiases = self.biases[:len(self.biases) - 1]
        tWeights = self.weights[:len(self.weights) - 1]

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

        # grab the last bias weight pair before ziping list together for iteration
        lBias = self.biases[-1]
        lWeight = self.weights[-1]
        
        # prepare all tanh/sigmoid weights in temp fields by removing the last item in each
        tBiases = self.biases[:len(self.biases) - 1]
        tWeights = self.weights[:len(self.weights) - 1]

        # stores the intermediate evaluation of the weights times previous layer activation,
        # prior to the activation step
        zList = []
        for b, w in zip(tBiases, tWeights):
            z = np.dot(w, activation) + b
            zList.append(z)
            activation = self.hypTangent(z)
            activationList.append(activation)
        
        # compute the final pass with the linear activation function
        z = np.dot(lWeight, activation) + lBias
        zList.append(z)
        activation = self.linearActivation(z)
        print(activation)
        activationList.append(activation)

        # backprop starts
        # traversing previous activation and intermediate lists in negative index as in the book
        # a key difference here is the derivative of the linear function, which is just zList[-1],
        # rather than the hypTanPrime(zList[-1]), as we're doing function approximation instead of
        # classification
        delta = self.costPrime(activationList[-1], y) * self.linearActivationPrime(zList[-1])
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
        return(dwrt_bias, dwrt_weight)

    '''
    Cost Function Derivative

    '''
    def costPrime(self, outputActivations, y):
        return(outputActivations-y)

    '''
    Finally, train the network using mini-batch training.
    Parameters are the batch and a learning rate (batch, learningRate), respectively.

    '''
    def train(self, batch, learningRate):
        dwrt_bias = [np.zeros(b.shape) for b in self.biases]
        dwrt_weight = [np.zeros(w.shape) for w in self.biases]

        for x in batch:
            size = int(len(x) - 1)
            indata = np.zeros((size, 1))
            for i in range(len(x) - 1):
                indata[i] = x[i]
            outdata = float(x[len(x) - 1])
            print(indata, outdata)
            delta_dwrt_bias, delta_dwrt_weight = self.backpropagate(indata, outdata)
            dwrt_bias = [db+ddb for db, ddb in zip(dwrt_bias, delta_dwrt_bias)]
            dwrt_weight = [dw+ddw for dw, ddw in zip(dwrt_weight, delta_dwrt_weight)]
        

        # update weight matrices after the batch is complete
        # adjustment amount is normalized by the batch size
        self.weights = [w - (learningRate/len(batch))*dw 
                            for w, dw in zip(self.weights, dwrt_weight)]
        self.biases = [b - (learningRate/len(batch))*db
                            for b, db in zip(self.biases, dwrt_bias)]


    '''
    Test method to find accuracy

    '''

    def test(self, testData):
        for x in testData:
            xval = np.zeros((len(x) - 1, 1))
            for i in range(len(x) - 1):
                xval[i] = x[i]
            yval = float(x[len(x) - 1])
            # print(xval)
            # test_results = [(np.argmax(self.forwardPass(xval)), yval)]
        # return sum(int(x == y) for (x, y) in test_results)
