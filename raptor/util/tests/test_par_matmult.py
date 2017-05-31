import numpy as np
import scipy as sp
from pyamg import smoothed_aggregation_solver as sas
from pyamg.gallery import diffusion_stencil_2d as diff
from pyamg.gallery import stencil_grid
from scipy.io import mmwrite
import sys

n = 75
eps = 0.001
theta = np.pi / 8.0
sten = diff(eps, theta)
A = stencil_grid(sten, (n,n))

ml = sas(A)

num_levels = len(ml.levels)
A = ml.levels[0].A
P = ml.levels[0].P
test_path = sys.argv[1]

# Remove any value with small magnitude from A
for i in range(A.nnz):
    val = A.data[i]
    if abs(val) <= 1e-15:
        A.data[i] = 0
A.eliminate_zeros()
mmwrite("%s/testA.mtx" %test_path, A)

# Remove any value with small magnitude from P
for i in range(P.nnz):
    val = P.data[i]
    if abs(val) <= 1e-15:
        P.data[i] = 0
P.eliminate_zeros()
mmwrite("%s/testP.mtx" %test_path, P)

# Calculate AP
AP = A*P

# Remove any value with small magnitude from AP
for i in range(AP.nnz):
    val = AP.data[i]
    if abs(val) <= 1e-15:
        AP.data[i] = 0
AP.eliminate_zeros()
mmwrite("%s/testAP.mtx" %test_path, AP)

# Calculate Ac with reduced P, AP
Ac = P.T * (AP)

# Remove any value with small magnitude from Ac
for i in range(Ac.nnz):
    val = Ac.data[i]
    if abs(val) <= 1e-15:
        Ac.data[i] = 0
Ac.eliminate_zeros()
mmwrite("%s/testAc.mtx" %test_path, Ac)




