import numpy as np

class RBFN(object):

    # ?????????????????
    #Do we want to hrad code sigma or make tunable?
    # ?????????????????
    # ?????????
    # Currently centers for basis functions are randomly chosen
    # Could do k-means(Sheppard said not necessary) or centers at regular intervals instead
    # ?????????
    # ???????????????
    # Currently the test method only displays the predicted outputs
    # could modify it to give detailed error stats
    # ???????????????

    def __init__(self, in_dim, basis_fxns, sigma = 1.0):
        self.in_dim = in_dim #in_dim: dimension of the input data
        self.basis_fxns = basis_fxns #basis_fxns: number of hidden radial basis functions
        self.sigma = sigma #Sigma is the spread of the basis function
        self.centers = None #Initially no centers for basis functions
        self.weights = None # initially no weights

    #This is the Gaussian basis function
    def _kernel_function(self, center, data_point):
        return np.exp((-np.linalg.norm(center-data_point)**2)/(2*(self.sigma**2)))

    """
    The below function:
    Calculates interpolation matrix, G, using self._kernel_function
    Where X is a numpy matrix of training data/vectors
    The interpolation matrix is an m*n matrix
    where m = (number of data points in x)
    and n = (number of basis functions)(i.e. hidden layer nodes)
    This matrix represents each basis function applied to each point in the training data
    This matrix is used to apply linear regression to update the weights
    """
    def _calculate_interpolation_matrix(self,X):
        #Creates an empty matrix to store interpolation of input data
        G = np.zeros((X.shape[0], self.basis_fxns))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                #The below calls the self._kernel_function, which applies the basis function
                # to each data point
                G[data_point_arg,center_arg] = self._kernel_function(center, data_point)
        return G

    """
     The train() method trains self.weights using linear regression
     Where X is a numpy matrix of the input training samples
     with dimensions (number of data samples)x(input dimensions
     And Y is a numpy matrix of the target output
     with dimensions (number of data samples)x(1)
    """
    def train(self,X,Y):
        #Below line creates a shuffled subset of vectors from the input, X
        #This random set of vectors becomes the centers for the basis functions
        random_args = np.random.permutation(X.shape[0]).tolist()
        # The below loop assigns centers for the number of basis functions
        self.centers = [X[arg] for arg in random_args][:self.basis_fxns]
        G = self._calculate_interpolation_matrix(X)
        #The pinv(G) is the pseudo-inverse of the interpolation matrix
        #The dot product of pinv(G) and Y yields the weights of the network
        self.weights = np.dot(np.linalg.pinv(G),Y)

    """
    The test() function takes a set of input vectors and predicts the output
    X is a numpy matrix of test data points
    dimensions of X are (number input samples)x(number input dimensions)
    """
    def test(self,X):
        G = self._calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights)
        return predictions
