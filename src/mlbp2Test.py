import numpy as np
import mlbp2

nn = mlbp2.mlbp2([4, 50, 1])

dataMatrix = np.loadtxt("data_4d.txt", delimiter=",")
# print(x[1:2])

nn.train(dataMatrix, 0.1)