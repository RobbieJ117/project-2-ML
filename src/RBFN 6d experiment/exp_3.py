import RBFN
import numpy as np
import numba
class Data_manipulator:
    def __init__(self, switch):
        in_use = self.load_data(switch)
        self.x, self.y = self.cleave(in_use)

    @numba.jit
    def load_data(self, switch):
        if switch == 0:
            return np.loadtxt("6D_0.txt", delimiter=",")  # read file into array
        elif switch == 1:
            return np.loadtxt("6D_1.txt", delimiter=",")  # read file into array
        elif switch == 2:
            return np.loadtxt("6D_2.txt", delimiter=",")  # read file into array
        elif switch == 3:
            return np.loadtxt("6D_3.txt", delimiter=",")  # read file into array
        elif switch == 4:
            return np.loadtxt("6D_4.txt", delimiter=",")  # read file into array
        elif switch == 5:
            return np.loadtxt("6D_5.txt", delimiter=",")  # read file into array
        elif switch == 6:
            return np.loadtxt("6D_6.txt", delimiter=",")  # read file into array
        elif switch == 7:
            return np.loadtxt("6D_7.txt", delimiter=",")  # read file into array
        elif switch == 8:
            return np.loadtxt("6D_8.txt", delimiter=",")  # read file into array
        elif switch == 9:
            return np.loadtxt("6D_9.txt", delimiter=",")  # read file into array
        else:
            return

    def print_validate(self):
        print("\nF1X\n")
        print(self.fold_1_x)
        print("\nF1Y\n")
        print(self.fold_1_y)
        print("\nF2X\n")
        print(self.fold_2_x)
        print("\nF2Y\n")
        print(self.fold_2_y)

    def cleave(self, in_matrix):
        x0 = in_matrix[:,0:-1]
        y0 = in_matrix[:,-1:]
        return x0, y0

    def reload(self, switch):
        if switch == 0:
            loaded = np.loadtxt("6D_0.txt", delimiter=",")  # read file into array
        elif switch == 1:
            loaded = np.loadtxt("6D_1.txt", delimiter=",")  # read file into array
        elif switch == 2:
            loaded = np.loadtxt("6D_2.txt", delimiter=",")  # read file into array
        elif switch == 3:
            loaded = np.loadtxt("6D_3.txt", delimiter=",")  # read file into array
        elif switch == 4:
            loaded = np.loadtxt("6D_4.txt", delimiter=",")  # read file into array
        elif switch == 5:
            loaded = np.loadtxt("6D_5.txt", delimiter=",")  # read file into array
        elif switch == 6:
            loaded = np.loadtxt("6D_6.txt", delimiter=",")  # read file into array
        elif switch == 7:
            loaded = np.loadtxt("6D_7.txt", delimiter=",")  # read file into array
        elif switch == 8:
            loaded = np.loadtxt("6D_8.txt", delimiter=",")  # read file into array
        elif switch==9:
            loaded = np.loadtxt("6D_9.txt", delimiter=",")  # read file into array
        else:
            return
        x0, y0 = self.cleave(loaded)
        self.x =x0
        self.y = y0



dm = Data_manipulator(0)




network = RBFN.RBFN("6D1", 6, 15)
for i in range(0, 10):
    holdout = i
    for j in range(0, 10):
        if holdout == j:
            print("nothing to do")
        else:
            dm.reload(j)
            network.train_wgts(dm.x, dm.y)
    dm.reload(holdout)
    network.test(dm.x, dm.y)
network.print_results()
