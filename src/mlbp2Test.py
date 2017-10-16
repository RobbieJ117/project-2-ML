import numpy as np
import mlbp2
from sklearn.preprocessing import normalize

nn = mlbp2.mlbp2([2, 50, 50, 1])


dataMatrix = np.loadtxt("data_2d.txt", delimiter=",")
dataMatrixNorm = dataMatrix
# print(x[1:2])
train = dataMatrixNorm[::2]
test = dataMatrixNorm[::1]


# print(train)

nn.stochGradientDescent(train, 50, 20, 0.01, test)
# nn.stochGradientDescent(test, 20, 100, 0.1, train)

# print(nn.weights, nn.biases)
# nn.stochGradientDescent(test, 200, 100, 0.01, train)


