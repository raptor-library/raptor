import scipy.io, PetscBinaryIO

def convert(file_in, file_out):
    A = scipy.io.mmread(file_in)
    A = A.tocsr()
    idxtype = '32bit'
    if (A.shape[0]) < (2**(21)-1):
        idxtype = '64bit'
    petsc_IO = PetscBinaryIO.PetscBinaryIO(indices=idxtype)
    petsc_IO.writeMatSciPy(open(file_out,'w'), A)
