import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os.path

class RBFN(object):


    def __init__(self, experiment_id, in_dim, basis_fxns):
        self.id = "{}.txt".format(experiment_id)
        self.in_dim = in_dim #in_dim: dimension of the input data
        self.basis_fxns = basis_fxns #basis_fxns: number of hidden radial basis functions
        self.centers = np.random.uniform(low=-3.0, high=3.0, size=(self.basis_fxns, self.in_dim-1))
        self.weights = np.random.uniform(low=-.1, high=.1, size=(basis_fxns, 1)) # initially random
        self.epochs = 100
        self.ada = .05
        self.sigma = 1 #Sigma is the spread of the basis function
        self.lossHistory = []
        self.iteration=1


    '''
    The below method takes N number of M-dimensional vectors (as the matrix X)
    for each input vector, each radial basis function is applied and put into the matrix G
    The rows of G represent an input vector and each column represents a radial basis function
    Therefore the element G[0,0] represents the first radial basis function applied to the input vector
    The dot product of the matrix G and the RBFN's weights yields the "results" matrix of dimensions (N)x(1)
    The results matrix represents each input after running through the network
    '''
    def feed_forward(self, X):
        G = np.zeros(len(X), self.basis_fxns)
        for data_point_arg in enumerate(X):
            for center_arg in enumerate(self.centers):
                # The below calls the self._kernel_function, which applies the basis function
                # to each data point
                G[data_point_arg, center_arg] = np.exp((-np.linalg.norm(self.centers[center_arg]-X[data_point_arg])**2)/(2*(self.sigma**2)))
        results = G.dot(self.weights)
        # return a matrix with dimensions: (num_basis_fxns)x(1)
        return results #This matrix represents the outputs for each input to the network


    """
     The train_wgts() method trains self.weights using gradient descent
     Where X is a numpy matrix of the input training samples
     with dimensions (number of data samples)x(input dimensions)
     And Y is a numpy matrix of the target output
     with dimensions (number of data samples)x(1)
    """
    def train_wgts(self,X,Y):
        self.grad_descent(X, Y)
        self.print_results(self.iteration)
        self.iteration+=1

    '''
    Chose to implement gradient descent as an auxiliary function called by train_wgts()
    '''
    def grad_descent(self, X, Y):
        for epoch in range(0, self.epochs):
            predicted = self.feed_forward(X)
            loss = 0.5*np.sum(np.square(predicted-Y))#calculate loss
            if (loss/(X.shape[0])) < 0.05: #If the average loss is smaller than .1 per input, break early
                return
            # divide by number of inputs to scale the gradients
            gradient = X.T.dot(loss) / X.shape[0]
            self.weights += (self.ada*gradient)#update the weights
            self.lossHistory.append(loss)


    """
    The test() function takes a set of input vectors and predicts the output
    X is a numpy matrix of test data points
    dimensions of X are (number input samples)x(number input dimensions)
    """
    def test(self, X, Y):
        i = self.iteration
        n = X.shape[0]
        G = self.feed_forward(X)
        rmse = np.sqrt(np.mean((X-Y))**2)#RMSE
        mae = np.mean(X-Y)#MAE
        A = np.hstack((X,G)) #set of vectors of predicted points
        B = np.hstack((X, Y)) #set of vectors for actual points
        res = 1 - np.dot(A / np.linalg.norm(A, axis=1)[..., None], (B / np.linalg.norm(B, axis=1)[..., None]).T)# compute cosine distance between vectors
        cos_dist = res.mean()# mean cosine distance
        reults_string = "\nIteration{}\n\nRMSE:{}\nMAE:{}\nMean Cosine similarity{}\n\n".format(i, rmse, mae, cos_dist)
        if not os.path.isfile(self.id):
            f = open(self.id, "w")
            header = "{}:\nAda:{}\nBasis functions:{}\nSigma:{}\n".format(self.id, self.ada, self.basis_fxns, self.sigma)
            f.write(header)
        else:
            f = open(self.id, "r+")
        f.write(reults_string)
        f.close()


    def print_results(self, iter):
        file_name = "{}_training_cycle{}.png".format(self.id, iter)
        fig = plt.figure()
        plt.plot(np.arange(0, self.epochs), self.lossHistory)
        fig.suptitle("Training Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        fig.savefig(file_name)
        plt.close(fig)
        self.lossHistory = []