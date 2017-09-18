import numpy as np
import scipy.sparse as sp
import random
from scipy.io import mmwrite
from scipy.sparse import csr_matrix

n_rows = 100
n_cols = 10
nnz_per_row = 4

adense = np.zeros((n_rows, n_cols))
random.seed()
for i in range(n_rows * nnz_per_row):
    row = random.randint(0, n_rows - 1)
    col = random.randint(0, n_cols - 1)
    adense[row, col] += random.random()

A = csr_matrix(adense)
mmwrite("../../build/raptor/tests/random.mtx", A)

x = np.ones((n_cols, 1))
b = A.dot(x)
np.savetxt("../../build/raptor/tests/random_ones_b.txt", b)
x = np.ones((n_rows, 1))
b = A.T.dot(x)
np.savetxt("../../build/raptor/tests/random_ones_b_T.txt", b)

x = np.arange(0, n_cols)
b = A.dot(x)
np.savetxt("../../build/raptor/tests/random_inc_b.txt", b)
x = np.arange(0, n_rows)
b = A.T.dot(x)
np.savetxt("../../build/raptor/tests/random_inc_b_T.txt", b)



