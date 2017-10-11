import numpy as np
import scipy.sparse as sp
from scipy.io import mmwrite
from pyamg.gallery.diffusion import diffusion_stencil_2d
from pyamg.gallery import stencil_grid
import pyamg

sten_27_lap = np.ones((3,3,3))
sten_27_lap *= -1
sten_27_lap[1,1,1] = 26
A_27_lap = stencil_grid(sten_27_lap, (10, 10, 10)).tocsr()
mmwrite("../../build/raptor/tests/laplacian27.mtx", A_27_lap)
outfile = open("../../build/raptor/tests/laplacian27_data.txt", "w")
outfile.write("%d %d\n" %(A_27_lap.shape[0], A_27_lap.shape[1]))
for i in range(0, A_27_lap.shape[0]):
    start = A_27_lap.indptr[i]
    end = A_27_lap.indptr[i+1]
    row_sum = 0.0
    for j in range(start, end):
        row_sum += A_27_lap.data[j]
    outfile.write("%d %d %lg\n" %(i, end - start, row_sum))
outfile.close()

eps = 0.001
theta = np.pi / 8.0
sten_aniso = diffusion_stencil_2d(epsilon=eps, theta=theta)
A_aniso = stencil_grid(sten_aniso, (25, 25)).tocsr()
mmwrite("../../build/raptor/tests/aniso.mtx", A_aniso)
outfile = open("../../build/raptor/tests/aniso_data.txt", "w")
outfile.write("%d %d\n" %(A_aniso.shape[0], A_aniso.shape[1]))
for i in range(0, A_aniso.shape[0]):
    start = A_aniso.indptr[i]
    end = A_aniso.indptr[i+1]
    row_sum = 0.0
    for j in range(start, end):
        row_sum += A_aniso.data[j]
    outfile.write("%d %d %lg\n" %(i, end - start, row_sum))
outfile.close()

