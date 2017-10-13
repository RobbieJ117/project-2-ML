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

'''
data_2d = open("data_2d.txt", 'w')
# data for 2
for x_1 in range(-102, 103):
    for x_2 in range(-102, 103):
        rS = rosenbrock([x_1/34, x_2/34])
        line = "%.2f,%.2f,%.2f\n" % (rS[0], rS[1], rS[2])
        line.strip()
        data_2d.write(line)
data_2d.close()

data_3d = open("data_3d.txt", 'w')
for x_1 in range(-30, 31):
    for x_2 in range(-30, 31):
        for x_3 in range(-30, 31):
            rS = rosenbrock([x_1/10, x_2/10, x_3/10])
            data_3d.write("%.2f,%.2f,%.2f,%.2f\n" % (rS[0], rS[1], rS[2], rS[3]))
data_3d.close()

data_4d = open("data_4d.txt", 'w')
for x_1 in range(-16, 17):
    for x_2 in range(-16, 17):
        for x_3 in range(-16, 17):
            for x_4 in range(-16, 17):
                rS = rosenbrock([x_1/4, x_2/4, x_3/4, x_4/4])
                data_4d.write("%.2f,%.2f,%.2f,%.2f,%.2f\n" % (rS[0], rS[1], rS[2], rS[3], rS[4]))
data_4d.close()

data_5d = open("data_5d.txt", 'w')
for x_1 in range(-9, 10):
    for x_2 in range(-9, 10):
        for x_3 in range(-9, 10):
            for x_4 in range(-9, 10):
                for x_5 in range(-9, 10):
                    rS = rosenbrock([x_1/3, x_2/3, x_3/3, x_4/3, x_5/3])
                    data_5d.write("%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n" % (rS[0], rS[1], rS[2], rS[3], rS[4], rS[5]))   
data_5d.close()

data_6d = open("data_6d.txt", 'w')
# data for 6
for x_1 in range(-6, 7):
    for x_2 in range(-6, 7):
        for x_3 in range(-6, 7):
            for x_4 in range(-6, 7):
                for x_5 in range(-6, 7):
                    for x_6 in range(-6, 7):
                        rS = rosenbrock([x_1/2, x_2/2, x_3/2, x_4/2, x_5/2, x_6/2])
                        data_6d.write("%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n" % (rS[0], rS[1], rS[2], rS[3], rS[4], rS[5], rS[6]))  
data_6d.close()


'''
If executed via command line takes arguments as list of floating point numbers and outputs list plus rosenbrock evaluation

'''



'''

I kept this below just in case we need it for some reason...

if __name__ == "__main__":
    if (len(sys.argv) > 2):                     # check for the correct number of arguments
        dimensions = len(sys.argv) - 1          # dimensions is determined by the length of the arguments - 1 as the first argument is the file name 
        vector = []                             # create list for the arguments
        for i in range(1, dimensions + 1):      # starting at index 1; dimensions increased to match length of arguments
            vector.append(float(sys.argv[i]))   # cast input to float before sending to rosenbrock()
        print(rosenbrock(dimensions, vector))
    else: 
        print("Error! Too few arguments. At least 2 are necessary, but {} received.".format(len(sys.argv) - 1))

'''