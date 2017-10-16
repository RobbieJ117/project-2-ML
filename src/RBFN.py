import numpy as np
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import math
import matplotlib.pyplot as plt
import os.path
import numba

'''
RBFN
Giovany Addun 2017
'''


class RBFN(object):

    def __init__(self, experiment_id, in_dim, basis_fxns):
        self.id = "{}.txt".format(experiment_id)
        self.in_dim = in_dim #in_dim: dimension of the input data
        self.basis_fxns = basis_fxns #basis_fxns: number of hidden radial basis functions
        self.centers = np.random.uniform(low=-3, high=3, size=(self.basis_fxns, self.in_dim))
        self.weights = np.random.uniform(low=-.1, high=.1, size=(basis_fxns, 1)) # initially random
        self.epochs = 40 # number of times the network compares a sample and applies gradient descent
        self.ada = .25 # a scalar for the corrections to be made to weights in the network
        self.sigma = 1.2 #Sigma is the spread of the basis function
        self.lossHistory = []
        self.iteration=1


    '''
    The below method takes N number of M-dimensional vectors (as the matrix X)
    for each input vector, each radial basis function is applied and put into the matrix G
    The rows of g represent an input vector and each column represents a radial basis function
    Therefore the element g[0,0] represents the first radial basis function applied to the first input vector
    The dot product of the matrix g and the RBFN's weights yields the output or predicted value from the network
    '''
    @numba.jit
    def feed_forward(self, X):
        g = np.empty((len(X), self.basis_fxns))
        for data_point_arg in range(0, len(X)):
            for center_arg in range(0, self.basis_fxns):
                # The below applies the basis function to each input vector
                g[data_point_arg, center_arg] = np.exp((-np.linalg.norm(self.centers[center_arg]-X[data_point_arg])**2)/(2*(self.sigma**2)))
        # The matrix g represents each basis function applied to each input vector
        # Where g[X,Y] is the Yth basis function applied to the Xth input
        # The dot product of g with the network's weights yields the predicted outputs for each input vector
        return g


    """
     The train_wgts() method trains self.weights using gradient descent
     Where X is a numpy matrix of the input training samples
     with dimensions (number of data samples)x(input dimensions)
     And Y is a numpy matrix of the target output
     with dimensions (number of data samples)x(1)
    """
    def train_wgts(self,X,Y):
        self.grad_descent(X, Y)
        self.iteration+=1

    '''
    Chose to implement gradient descent as an auxiliary function called by train_wgts()
    This function takes a set of inputs X, predicts their outputs, compares that prediction to the actual outputs
    calculates the error
    '''
    def grad_descent(self, X, Y):
        for i in range(0, self.epochs):
            print("epoch{}".format(i))
            #The "inputs by functions" matrix is the same matrix g from the
            #feed_forward() method
            input_x_fxn_matrix = self.feed_forward(X)
            #The matrix of predicted outputs is calculated by taking the dot product
            #of the "inputs by functions" matrix and the weights of the network
            predicted = input_x_fxn_matrix.dot(self.weights)
            error = (predicted-Y)
            loss = np.sum(np.square(error))
            self.lossHistory.append(loss)
            # divide by number of inputs to scale the gradients
            gradient = input_x_fxn_matrix.T.dot(error)
            gradient = -1*self.ada*gradient/X.shape[0]
            self.weights += gradient#update the weights




    """
    The test() function takes a set of input vectors and predicts the output
    X is a numpy matrix of test data points
    dimensions of X are (number input samples)x(number input dimensions)
    And Y is the actual values of the Rosenbrock function at the points X
    """
    def test(self, X, Y):
        i = self.iteration-1
        G = self.feed_forward(X).dot(self.weights)
        mae = metrics.mean_absolute_error(Y, G)
        rmse = metrics.mean_squared_error(Y, G)
        rmse = math.sqrt(rmse)
        mean_y = np.mean(Y)
        mean_g = G[10:20]
        A = np.hstack((X,G)) #set of vectors of predicted points
        B = np.hstack((X,Y)) #set of vectors for actual points
        res = 1 - np.dot(A / np.linalg.norm(A, axis=1)[..., None], (B / np.linalg.norm(B, axis=1)[..., None]).T)# compute cosine distance between vectors
        cos_dist = res.mean()# mean cosine distance
        reults_string = "\nIteration{}\n\nRMSE:{}\nMAE:{}\nMean Cosine similarity{}\nMean Y{}\n Mean G{}\n".format(i, rmse, mae, cos_dist, mean_y, mean_g)
        if not os.path.isfile(self.id):
            f = open(self.id, "w")
            header = "{}:\nAda:{}\nEpochs:{}\nBasis functions:{}\nSigma:{}\n".format(self.id, self.ada, self.epochs, self.basis_fxns, self.sigma)
            f.write(header)
        else:
            f = open(self.id, "a")
        f.write(reults_string)
        f.close()


    '''
    Prints a plot of the error versus training iterations
    '''
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
