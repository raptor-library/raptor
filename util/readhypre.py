import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

class hypre_mat:
    def __init__(self, fname, rank):
        self.rank = rank
        self.read_info(fname)
        self.diag = self.read_csr(fname, 'D')
        self.offd = self.read_csr(fname, 'O', self.diag.shape)

    def read_info(self, fname):
        rank = self.rank
        with open(f'{fname}.INFO.{rank}') as fh:
            self.global_nrows = eval(fh.readline()[:-1])
            self.global_ncols = eval(fh.readline()[:-1])
            ncols_offd = eval(fh.readline()[:-1])
            ranges = fh.readline()[:-1].split()
            self.row_rng = (eval(ranges[0]), eval(ranges[1]) - 1)
            self.col_rng = (eval(ranges[2]), eval(ranges[3]) - 1)
            self.colmap_offd = np.zeros(ncols_offd, dtype=int)
            for i in range(ncols_offd):
                self.colmap_offd[i] = eval(fh.readline()[:-1])

    def read_csr(self, fname, suff, shape = None):
        try:
            with open(f'{fname}.{suff}.{self.rank}') as fh:
                num_rows = eval(fh.readline()[:-1])
                rowptr = np.zeros(num_rows + 1, dtype=int)
                for i in range(num_rows + 1):
                    rowptr[i] = eval(fh.readline()[:-1]) - 1
                colind = np.zeros(rowptr[num_rows], dtype=int)
                values = np.zeros(rowptr[num_rows], dtype=np.float64)
                for i in range(rowptr[num_rows]):
                    colind[i] = eval(fh.readline()[:-1]) - 1
                for i in range(rowptr[num_rows]):
                    values[i] = eval(fh.readline()[:-1])
                return csr_matrix((values, colind, rowptr))
        except FileNotFoundError:
            return csr_matrix(shape)

def merge_ranks(mats):
    rowptr = np.zeros(mats[0].global_nrows + 1, dtype=int)
    for mat in mats:
        for r in range(mat.diag.indptr.shape[0] - 1):
            rowptr[mat.row_rng[0] + r + 1] = (rowptr[mat.row_rng[0] + r] +
                                              (mat.diag.indptr[r + 1] - mat.diag.indptr[r]) +
                                              (mat.offd.indptr[r + 1] - mat.offd.indptr[r]))
    colind = np.zeros(rowptr[-1], dtype=int)
    values = np.zeros(rowptr[-1], dtype=np.float64)
    curr = 0
    for mat in mats:
        for r in range(mat.diag.indptr.shape[0] - 1):
            rng = (mat.diag.indptr[r], mat.diag.indptr[r+1])
            nnz = rng[1] - rng[0]
            colind[curr:curr + nnz] = mat.diag.indices[rng[0]:rng[1]] + mat.col_rng[0]
            values[curr:curr + nnz] = mat.diag.data[rng[0]:rng[1]]
            curr += nnz
            rng = (mat.offd.indptr[r], mat.offd.indptr[r+1])
            nnz = rng[1] - rng[0]
            colind[curr:curr + nnz] = mat.colmap_offd[mat.offd.indices[rng[0]:rng[1]]]
            values[curr:curr + nnz] = mat.offd.data[rng[0]:rng[1]]
            curr += nnz

    return csr_matrix((values, colind, rowptr))

def readij(base):
    import glob
    nprocs = len(glob.glob(f'{base}.*'))
    def parse_hypre(fname):
        data = np.loadtxt(fname, skiprows=1)
        return [np.array(data[:, 0], dtype=int),
                np.array(data[:, 1], dtype=int),
                np.array(data[:, 2], dtype=np.float64)]

    fnames = [f'{base}.{"0" * (5 - len(str(r))) + str(r)}' for r in range(nprocs)]
    proc_data = [parse_hypre(fname) for fname in fnames]
    I = np.concat([p[0] for p in proc_data])
    J = np.concat([p[1] for p in proc_data])
    data = np.concat([p[2] for p in proc_data])
    return coo_matrix((data, (I, J)))

