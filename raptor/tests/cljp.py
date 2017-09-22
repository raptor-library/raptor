import numpy as np
import scipy.sparse as sp
import random
from scipy.io import mmwrite
from scipy.sparse import csr_matrix
from pyamg.gallery.stencil import stencil_grid
from pyamg.gallery.diffusion import diffusion_stencil_2d
from pyamg.classical import ruge_stuben_solver as rss
from pyamg.strength import classical_strength_of_connection
from pyamg.util.utils import remove_diagonal

# Form RSS laplacian hierarchy
sten = np.ones((3,3,3))
sten *= -1
sten[1,1,1] = 26
A = stencil_grid(sten, (5, 5, 5)).tocsr()

# Get strength and transpose
S = classical_strength_of_connection(A, 0.25)
S = remove_diagonal(S)
T = S.T.tocsr()

# Determine CLJP splitting
splitting = np.empty(S.shape[0], dtype = 'intc')
weight = np.empty(S.shape[0])

# Initialize weights
for i in range(S.shape[0]):
    splitting[i] = -1;
    weight[i] = ((float)(i)) / S.shape[0];
unassigned = S.shape[0]
for i in range(S.shape[0]):
    for jj in range(S.indptr[i], S.indptr[i+1]):
        j = S.indices[jj]
        if (j != i):
            weight[j] += 1

edgemark = np.ones((S.nnz,))
dep_c_cache = np.zeros((T.shape[0],))

while(unassigned):
    coarse_list = []
    for i in range(S.shape[0]):
        to_coarse = 0
        if splitting[i] == -1:
            to_coarse = 1
            for jj in range(S.indptr[i], S.indptr[i+1]):
                j = S.indices[jj]
                if splitting[j] == -1 and weight[j] > weight[i]:
                    to_coarse = 0
                    break;
            if to_coarse:
                for jj in range(T.indptr[i], T.indptr[i+1]):
                    j = T.indices[j]
                    if splitting[j] == -1 and weight[j] > weight[i]:
                        to_coarse = 0
                        break


            if to_coarse == 1:
                coarse_list.append(i)
                unassigned -= 1

    for c in coarse_list:
        splitting[c] = 1


    for c in coarse_list:
        for jj in range(S.indptr[c], S.indptr[c+1]):
            j = S.indices[jj]
            if splitting[j] == -1 and edgemark[jj] != 0:
                edgemark[jj] = 0
                weight[j] -= 1
                if (weight[j] < 1):
                    splitting[j] = 0
                    unassigned -= 1

    for c in coarse_list:
        for jj in range(T.indptr[c], T.indptr[c+1]):
            j = T.indices[jj]
            if splitting[j] == -1:
                dep_c_cache[j] = 1
        for jj in range(T.indptr[c], T.indptr[c+1]):
            j = T.indices[jj]
            for kk in range(S.indptr[j], S.indptr[j+1]):
                k = S.indices[kk]
                if splitting[k] == -1 and edgemark[kk] != 0:
                    if dep_c_cache[k] == 1:
                        edgemark[kk] = 0;
                        weight[k] -= 1
                        if (weight[k] < 1):
                            splitting[k] = 0
                            unassigned -= 1


from pyamg.classical import split
splitting_pyamg = split.CLJP(S, color=True)
for i in range(S.shape[0]):
    print splitting[i], splitting_pyamg[i]
