import Mlbp
import numpy as np
iV = np.matrix('1 2 3 5 8 1')
nn = Mlbp.Mlbp(6, 1, 100, 1, 0.1)
nn.train(iV, 10)