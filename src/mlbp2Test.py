import numpy as np
import mlbp2
from sklearn.preprocessing import normalize



class Data_manipulator:
    def __init__(self, path):
        all_data=np.loadtxt(path, delimiter=",") #read file into ndarray
        # all_data= self.standardize(all_data)
        self.n = len(all_data)#Total number of samples
        self.h =  self.n//2#Half the samples rounded down
        fold_1, fold_2 = self.shuffle_fold(all_data)
        self.fold_1_x, self.fold_1_y = self.cleave(fold_1)# creates input vectors and targets for fold 1
        self.fold_2_x, self.fold_2_y = self.cleave(fold_2)# creates input vectors for targets in fold 2
        
    #I put this in here for testing
    def print_validate(self):
        print("\nF1X\n")
        print(self.fold_1_x)
        print("\nF1Y\n")
        print(self.fold_1_y)
        print("\nF2X\n")
        print(self.fold_2_x)
        print("\nF2Y\n")
        print(self.fold_2_y)

    #Randomly reorders rows from input data
    #separates into top half and bottom half of rows
    def shuffle_fold(self, input_data):
        np.random.shuffle(input_data)
        f_1 = input_data[0:self.h, :]
        start_i = (self.h)
        end_i = (self.n)
        f_2 = input_data[start_i:-1, :]
        return f_1, f_2
    
    #separates the input vectors from their target outputs
    def cleave(self, in_matrix):
        x = in_matrix[:,0:-1]
        y = in_matrix[:,-1:]
        return x, y
    
    #recombines previously sampled data into new samples
    def recombine(self):
        fold_1 = np.hstack((self.fold_1_x, self.fold_1_y))
        fold_2 = np.hstack((self.fold_2_x, self.fold_2_y))
        whole_data = np.vstack((self.fold_1, self.fold_2))
        self.fold_1, self.fold_2 = self.shuffle_fold(whole_data)
        self.fold_1_x, self.fold_1_y = self.cleave(fold_1)
        self.fold_2_x, self.fold_2_y = self.cleave(fold_2)
        # self.print_validate()

    def standardize(self, in_matrix):
        in_matrix[:, 1] = (in_matrix[:, 1]-np.mean(in_matrix[:, 1])/np.std(in_matrix[:, 1]))
        return in_matrix


# '''
# Example usage
# '''
# dm = Data_manipulator("data_2D.txt")
# dm.print_validate()
# network = RBFN.RBFN("2d", 2, 80)
# # This loop executes 5x2 cross validation
# for i in range(0, 5):
#     print('started training{}'.format(i))
#     network.train_wgts(dm.fold_1_x, dm.fold_1_y)
#     network.test(dm.fold_2_x, dm.fold_2_y)
#     dm.recombine()
# network.print_results()


# MLBP Test Script
dm = Data_manipulator("data_linear.txt")

# standardize input data
# dm.fold_1_x = dm.standardize(dm.fold_1_x)
# dm.fold_2_x = dm.standardize(dm.fold_2_x)

# put back together for input requirements
testset_1 = np.hstack((dm.fold_1_x, dm.fold_1_y))
testset_2 = np.hstack((dm.fold_2_x, dm.fold_2_y))

dm.print_validate()


network = mlbp2.mlbp2([1, 4, 4, 1], 'linear')


network.stochGradientDescent(testset_1, 100, 5, 0.0000000015, testset_2)
