import numpy as np


def linearFunction(x):
    return np.sin(x) + 2

dataLinear = open('data_linear.txt', 'w')
for i in range(5000):
    dataLinear.write("{},{}\n".format(i, linearFunction(i)))
