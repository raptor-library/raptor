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
x = np.ones((A_27_lap.shape[0], 1))
b = A_27_lap.dot(x)
np.savetxt("../../build/raptor/tests/laplacian27_ones_b.txt", b);
b = A_27_lap.T.dot(x)
np.savetxt("../../build/raptor/tests/laplacian27_ones_b_T.txt", b);
for i in range(0, A_27_lap.shape[0]):
    x[i] = i
b = A_27_lap.dot(x)
np.savetxt("../../build/raptor/tests/laplacian27_inc_b.txt", b);
b = A_27_lap.T.dot(x)
np.savetxt("../../build/raptor/tests/laplacian27_inc_b_T.txt", b);


eps = 0.001
theta = np.pi / 8.0
sten_aniso = diffusion_stencil_2d(epsilon=eps, theta=theta)
A_aniso = stencil_grid(sten_aniso, (25, 25)).tocsr()
x = np.ones((A_aniso.shape[0], 1))
b = A_aniso.dot(x)
np.savetxt("../../build/raptor/tests/aniso_ones_b.txt", b);
b = A_aniso.T.dot(x)
np.savetxt("../../build/raptor/tests/aniso_ones_b_T.txt", b);
for i in range(0, A_aniso.shape[0]):
    x[i] = i
b = A_aniso.dot(x)
np.savetxt("../../build/raptor/tests/aniso_inc_b.txt", b);
b = A_aniso.T.dot(x)
np.savetxt("../../build/raptor/tests/aniso_inc_b_T.txt", b);


