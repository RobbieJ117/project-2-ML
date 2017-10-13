import sys
import csv
import numpy as np


'''
Rosenbrock takes an integer dimensionality and a list of floating point numbers as arguments.
It returns the original vector, with the summation appended as the last element.
'''

def rosenbrock(dimension, vector):
    sum = 0
    for i in range(0, dimension - 2):
        sum += ((((vector[i]*vector[i])-vector[i+1])**2)+(vector[i]-1)**2)
    vector.append(sum)
    return vector

'''
evalutes the rosenbrock function and prints to a csv file
Note from Robbie: I got it to write to a csv file which took some time to understand, and I implemented
the loops. I am not 100% sure on some of the values to enter into the rosenbrock function.
'''
writer_d6 = open("dataGenerated_6D.csv", 'w')
writer_d5 = open("dataGenerated_5D.csv", 'w')
writer_d4 = open("dataGenerated_4D.csv", 'w')
writer_d3 = open("dataGenerated_3D.csv", 'w')
result3 = np.arr
for x_1 in range(-3, 3):
    for x_2 in range(-3, 3):
        result3 = rosenbrock(3, [x_1, x_2])
        np.savetxt("dataGenerated_3D.csv", result, delimiter=",")
        for x_3 in range(-3, 3):
            result4 = rosenbrock(4, [x_1, x_2, x_3])
            np.savetxt("dataGenerated_4D.csv", result, delimiter=",")
            for x_4 in range(-3, 3):
                result5 = rosenbrock(5, [x_1, x_2, x_3, x_4])
                np.savetxt("dataGenerated_5D.csv", result, delimiter=",")
                for x_5 in range(-3, 3):
                    result6 = rosenbrock(6, [x_1, x_2, x_3, x_4, x_5])
                    np.savetxt("dataGenerated_6D.csv", result, delimiter=",")
                    x_5=x_5+0.3
                x_4 = x_4 + 0.3
            x_3 = x_3 + 0.3
        x_2 = x_2 + 0.3
    x_1 = x_1 + 0.3
    writer_d6.close()
    writer_d5.close()
    writer_d4.close()
    writer_d3.close()



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