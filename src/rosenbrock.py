import sys
import csv
import numpy as np


'''
Rosenbrock takes an integer dimensionality and a list of floating point numbers as arguments.
It returns the original vector, with the summation appended as the last element.

'''
def rosenbrock(dimension, vector):
    sum = 0
    for i in range(0, dimension - 1):
        sum += pow((1 - vector[i]), 2) + (100 * pow((vector[i+1] - pow(vector[i], 2)), 2))
    vector.append(sum)
    return vector

'''
evalutes the rosenbrock function and prints to a csv file
Note from Robbie: I got it to write to a csv file which took some time to understand, and I implemented
the loops. I am not 100% sure on some of the values to enter into the rosenbrock function.
'''
writer = csv.writer(open("dataGenerated.csv", 'w'))
writer.writerow( ('X2', ' X3', ' X4', ' X5', ' X6', ' SUM') )
for x_1 in range(-3, 3):
    for x_2 in range(-3, 3):
        for x_3 in range(-3, 3):
            for x_4 in range(-3, 3):
                for x_5 in range(-3, 3):
                    for x_6 in range(-3, 3):
                        x_6 = rosenbrock(5, [x_2, x_3, x_4, x_5, x_6])
                        writer.writerow(x_6)


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