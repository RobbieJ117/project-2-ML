import numpy as np
import math
import matplotlib.pyplot as plt

class RBFN(object):

    def __init__(self, in_dim, basis_fxns, sigma = 1.0):
        self.in_dim = in_dim #in_dim: dimension of the input data
        self.basis_fxns = basis_fxns #basis_fxns: number of hidden radial basis functions
        self.sigma = sigma #Sigma is the spread of the basis function
        self.centers = np.random.uniform(low=-3.0, high=3.0, size=(self.basis_fxns, self.in_dim-1))
        self.weights = np.random.uniform(low=-.1, high=.1, size=(basis_fxns, 1)) # initially random
        self.epochs = 100
        self.ada = .05
        self.lossHistory = []


    #This is the Gaussian basis function
    def feed_forward(self, X):
        G = np.zeros(len(X), self.basis_fxns)
        for data_point_arg in enumerate(X):
            for center_arg in enumerate(self.centers):
                # The below calls the self._kernel_function, which applies the basis function
                # to each data point
                G[data_point_arg, center_arg] = np.exp((-np.linalg.norm(self.centers[center_arg]-X[data_point_arg])**2)/(2*(self.sigma**2)))
                G = G.dot(self.weights)
                # return a matrix with dimensions: (num_basis_fxns)x(1)
        return G #This matrix represents the outputs for each input to the network


    """
     The train() method trains self.weights using linear regression
     Where X is a numpy matrix of the input training samples
     with dimensions (number of data samples)x(input dimensions)
     And Y is a numpy matrix of the target output
     with dimensions (number of data samples)x(1)
    """
    def train(self,X,Y):



    def grad_descent(self, X, Y):
        for epoch in range(0, self.epochs):
            predicted = self.feed_forward(X)
            loss = np.sum((predicted-Y)**2)#calculate loss
            # divide by number of inputs to scale the gradients
            gradient = X.T.dot(error) / X.shape[0]
            self.weights += (self.ada*gradient)#update the weights
            self.lossHistory.append(loss)




    """
    The test() function takes a set of input vectors and predicts the output
    X is a numpy matrix of test data points
    dimensions of X are (number input samples)x(number input dimensions)
    """
    def test(self, X, Y):
        G = self.activation_fxn(X)
        predictions = np.dot(G, self.weights)
        return predictions
