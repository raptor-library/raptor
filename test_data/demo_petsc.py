import pyamg
import scipy.io, PetscBinaryIO

A = pyamg.gallery.poisson((5,5), format='csr')
PetscBinaryIO.PetscBinaryIO().writeMatSciPy(open('mat.pm','w'), A)
