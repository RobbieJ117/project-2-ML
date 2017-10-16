import numpy as np
import mlbp2

nn = mlbp2.mlbp2([6, 1000, 25, 1])

dataMatrix = np.loadtxt("data_6d.txt", delimiter=",")
# print(x[1:2])
inputList = []
checkList = []
for x in dataMatrix:
    size = int(len(x) - 1)
    indata = np.zeros((size, 1))
    for i in range(len(x) - 1):
        indata[i] = x[i]
    outdata = float(x[len(x) - 1])
    inputList.append(indata)
    checkList.append(outdata)


nn.stochGradientDescent(dataMatrix, 20, 10, 0.01, dataMatrix)


