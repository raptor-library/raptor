import numpy as np
import sys
from scipy.sparse import bsr_matrix, csr_matrix, coo_matrix

from enum import Enum

class mat_type(Enum):
    ParCSR = 0
    ParBSR = 1
    BSR = 2
    CSR = 3

def isblock(mtype):
    return mtype is mat_type.BSR or mtype is mat_type.ParBSR

def ispar(mtype):
    return mtype is mat_type.ParCSR or mtype is mat_type.ParBSR

class mat_header:
    def __init__(self, arr):
        assert(arr.shape[0] >= 3 and arr.shape[0] <= 6)
        pos = 0
        self.mat_type = mat_type(arr[pos])
        pos += 1
        self.shape = (arr[pos], arr[pos+1])
        pos += 2
        if ispar(self.mat_type):
            self.nprocs = arr[3]
            pos += 1
        else:
            self.nprocs = 1
        if isblock(self.mat_type):
            self.block_shape = (arr[pos], arr[pos+1])


def parse(fname, b_size):
        offset = 0
        nrows = np.fromfile(fname, dtype='int32', count=1)[0]
        offset += 4
        rowptr = np.fromfile(fname, dtype='int32', offset=offset, count=nrows+1)
        offset += 4 * rowptr.shape[0]
        nnz = rowptr[-1]
        colind = np.fromfile(fname, dtype='int32', offset=offset, count=nnz)
        offset += 4 * colind.shape[0]
        values = np.fromfile(fname, dtype=np.double, offset=offset, count=nnz*b_size)
        return [rowptr, colind, values]

def read(base):
    info = mat_header(np.fromfile(f'{base}.hdr', dtype='int32'))
    print(f'{info.mat_type} on {info.nprocs} ranks')

    nprocs = info.nprocs
    b_size = np.prod(info.block_shape) if isblock(info.mat_type) else 1

    if ispar(info.mat_type):
        proc_data = [parse(f'{base}.{r}', b_size) for r in range(nprocs)]

        nrows = sum([p[0].shape[0] - 1 for p in proc_data])
        nnz = sum([p[1].shape[0] for p in proc_data])

        for r in range(nprocs - 1):
            proc_data[r+1][0] = proc_data[r+1][0] + proc_data[r][0][-1]

        rowptr = np.concat([proc_data[i][0] if i == 0 else proc_data[i][0][1:] for i in range(nprocs)])
        colind = np.concat([p[1] for p in proc_data])

        values = np.concat([p[2] for p in proc_data])
        if info.mat_type is mat_type.ParBSR:
            values = values.reshape(nnz, info.block_shape[0], info.block_shape[1])
            return bsr_matrix((values, colind, rowptr))
        elif info.mat_type is mat_type.ParCSR:
            return csr_matrix((values, colind, rowptr))
    else:
        data = parse(base, b_size)
        if info.mat_type is mat_type.BSR:
            data[2] = data[2].reshape(data[1].shape[0], info.block_shape[0], info.block_shape[1])
            print(data[0])
            return bsr_matrix((data[2], data[1], data[0]))
        else:
            return csr_matrix((data[2], data[1], data[0]))

on_on = read(full('C_on_on'))
a = read(full('onproc'))
