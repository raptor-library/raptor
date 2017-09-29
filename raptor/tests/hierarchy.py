import numpy as np
import scipy.sparse as sp
import random
from scipy.io import mmwrite
from scipy.sparse import csr_matrix
from pyamg.gallery.stencil import stencil_grid
from pyamg.gallery.diffusion import diffusion_stencil_2d
from pyamg.classical import ruge_stuben_solver as rss

# Form RSS laplacian hierarchy
sten = np.ones((3,3,3))
sten *= -1
sten[1,1,1] = 26
A = stencil_grid(sten, (25, 25, 25)).tocsr()
ml = rss(A, keep = True, strength=('classical',{'theta' : 0.25}), CF='CLJP')
for i in range(0, len(ml.levels)):
    level = ml.levels[i]
    A = level.A
    mmwrite("../../build/raptor/tests/rss_laplace_A%d" %i, level.A)
    if (i < len(ml.levels) - 1):
        mmwrite("../../build/raptor/tests/rss_laplace_P%d" %i, level.P)
        mmwrite("../../build/raptor/tests/rss_laplace_S%d" %i, level.C)
        np.savetxt("../../build/raptor/tests/rss_laplace_cf%d.txt" %i,
                level.splitting, fmt="%d")

    

eps = 0.001
theta = np.pi / 8.0
sten = diffusion_stencil_2d(epsilon=eps, theta=theta)
A = stencil_grid(sten, (50, 50)).tocsr()
ml = rss(A, keep = True, strength=('classical',{'theta' : 0.0}), CF='CLJP')
for i in range(0, len(ml.levels)):
    level = ml.levels[i]
    mmwrite("../../build/raptor/tests/rss_aniso_A%d" %i, level.A)
    if (i < len(ml.levels) - 1):
        mmwrite("../../build/raptor/tests/rss_aniso_P%d" %i, level.P)
        mmwrite("../../build/raptor/tests/rss_aniso_S%d" %i, level.C)
        np.savetxt("../../build/raptor/tests/rss_aniso_cf%d.txt" %i,
                level.splitting, fmt="%d")


