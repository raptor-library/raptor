def strength(A, theta = 0.0):
    A = A.tocsr()
    
    Sp = np.empty_like(A.indptr)
    Sj = np.empty_like(A.indices)
    Sx = np.empty_like(A.data)
    
    nnz = 0
    n = A.shape[0]
    
    Sp[0] = 0
    for i in range(0, n):
        diag = 0
        for j in range(A.indptr[i], A.indptr[i+1]):
            if (A.indices[j] == i):
                diag = A.data[j]
                break
        
        if diag < 0:
            row_scale = sys.minint
            for j in range(A.indptr[i], A.indptr[i+1]):
                val = A.data[j]
                if (val > row_scale):
                    row_scale = val
        else:
            row_scale = sys.maxint
            for j in range(A.indptr[i], A.indptr[i+1]):
                val = A.data[j]
                if (val < row_scale):
                    row_scale = val
        
        threshold = row_scale * theta
        
        Sj[nnz] = i
        Sx[nnz] = diag
        nnz += 1
        
        if diag < 0:
            for j in range(A.indptr[i], A.indptr[i+1]):
                val = A.data[j]
                if val > threshold:
                    col = A.indices[j]
                    Sj[nnz] = col
                    Sx[nnz] = val
                    nnz += 1
        else:
            for j in range(A.indptr[i], A.indptr[i+1]):
                val = A.data[j]
                if val < threshold:
                    col = A.indices[j]
                    Sj[nnz] = col
                    Sx[nnz] = val
                    nnz += 1
                    
        Sp[i + 1] = nnz

    return csr_matrix((Sx, Sj, Sp), shape=A.shape)
