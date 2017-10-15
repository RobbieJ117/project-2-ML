import sys
import decimal
import numpy as np


'''
Rosenbrock takes an integer dimensionality and a list of floating point numbers as arguments.
It returns the original vector, with the summation appended as the last element.

'''
def rosenbrock(vector):
    sum = 0
    for i in range(0, len(vector) - 1):
        sum += pow((1 - vector[i]), 2) + (100 * pow((vector[i+1] - pow(vector[i], 2)), 2))
    vector.append(sum)
    return vector

'''
The following produces 5 seperate data files for dimensions of n = {2, 3, 4, 5, 6} as comma seperated .txt files
For n = {2, 3, 4} the intervals between points of Rosenbrock inputs are 0.42
For n = {5, 6}, the intervals between inputs are 0.6
'''

data_2d = open("data_2d.txt", 'w')
data_3d = open("data_3d.txt", 'w')
data_4d = open("data_4d.txt", 'w')
x_1 = -2.2
while x_1 <2.1:
    x_2 = -2.2
    while x_2 < 2.1:
        rS = rosenbrock([x_1, x_2])
        data_2d.write("{},{},{}\n".format(rS[0], rS[1], rS[2]))
        x_3 = -2.2
        while x_3 < 2.1:
            rS = rosenbrock([x_1, x_2, x_3])
            data_3d.write("{},{},{},{}\n".format(rS[0], rS[1], rS[2], rS[3]))
            x_4 = -2.2
            while x_4 < 2.1:
                rS = rosenbrock([x_1, x_2, x_3, x_4])
                data_4d.write("{},{},{},{},{}\n".format(rS[0], rS[1], rS[2], rS[3], rS[4]))
                x_4+=.42
            x_3+=.42
        x_2+=.42
    x_1+=.42
data_2d.close()
data_3d.close()
data_4d.close()


'''
Create data for larger dimensionalities
'''


data_5d = open("data_5d.txt", 'w')
data_6d = open("data_6d.txt", 'w')
x_1 = -2.2
while x_1 <2.1:
    x_2 = -2.2
    while x_2 < 2.1:
        x_3 = -2.2
        while x_3 < 2.1:
            x_4 = -2.2
            while x_4 < 2.1:
                x_5 = -2.2
                while x_5 < 2.1:
                    rS = rosenbrock([x_1, x_2, x_3, x_4, x_5])
                    data_5d.write("{},{},{},{},{},{}\n".format(rS[0], rS[1], rS[2], rS[3], rS[4], rS[5]))
                    x_6 = -2.2
                    while x_6 < 2.1:
                        rS = rosenbrock([x_1, x_2, x_3, x_4, x_5, x_6])
                        data_6d.write("{},{},{},{},{},{},{}\n".format(rS[0], rS[1], rS[2], rS[3], rS[4], rS[5], rS[6]))
                        x_6+=.525
                    x_5+=.525
                x_4+=.525
            x_3+=.525
        x_2+=.525
    x_1+=.525
data_5d.close()
data_6d.close()
