class Data_manipulator:
    def __init__(self, path):
        all_data=np.loadtxt(path, delimiter=",") #read file into array
        self.n = len(all_data)
        self.h =  self.n//2
        fold_1, fold_2 = self.shuffle_fold(all_data)
        self.fold_1_x, self.fold_1_y = self.cleave(fold_1)
        self.fold_2_x, self.fold_2_y = self.cleave(fold_2)

    def print_validate(self):
        print("\nF1X\n")
        print(self.fold_1_x)
        print("\nF1Y\n")
        print(self.fold_1_y)
        print("\nF2X\n")
        print(self.fold_2_x)
        print("\nF2Y\n")
        print(self.fold_2_y)

    def shuffle_fold(self, input_data):
        np.random.shuffle(input_data)
        f_1 = input_data[0:self.h, :]
        start_i = (self.h)
        end_i = (self.n)
        f_2 = input_data[start_i:-1, :]
        return f_1, f_2

    def cleave(self, in_matrix):
        x = in_matrix[:,0:-1]
        y = in_matrix[:,-2:-1]
        return x, y

    def recombine(self):
        fold_1 = np.hstack((self.fold_1_x, self.fold_1_y))
        fold_2 = np.hstack((self.fold_2_x, self.fold_2_y))
        whole_data = np.vstack((fold_1, fold_2))
        fold_1, fold_2 = self.shuffle_fold(whole_data)
        self.fold_1_x, self.fold_1_y = self.cleave(fold_1)
        self.fold_2_x, self.fold_2_y = self.cleave(fold_2)
        self.print_validate()

dm = Data_manipulator("data_2D.txt")
print(dm.fold_1_x.shape)
print(dm.fold_1_y.shape)
print(dm.fold_2_x.shape)
print(dm.fold_2_y.shape)
dm.print_validate()
network = RBFN.RBFN("2d_4", 2, 80)
for i in range(0, 5):
    print('started training{}'.format(i))
    network.train_wgts(dm.fold_1_x, dm.fold_1_y)
    network.test(dm.fold_2_x, dm.fold_2_y)
    dm.recombine()
network.print_results()