def interp(A, S, splitting):
    n = A.shape[0]
    Pp = np.empty_like(A.indptr)
    
    S = S.copy()
    S.data[:] = 1.0
    S = S.multiply(A)
    AT = A.tocsc()
    
    A.sort_indices()
    AT.sort_indices()
    S.sort_indices()
    
    # Find shape of P (nnz per row)
    nnz = 0
    Pp[0] = 0
    for i in range(n):
        if splitting[i] == 1:
            nnz += 1
        else:
            for jj in range(S.indptr[i], S.indptr[i+1]):
                col = S.indices[jj]
                if ((splitting[col] == 1) and (col != i)):
                    nnz += 1
        Pp[i+1] = nnz
                               
    Pj = np.empty(nnz, dtype=Pp.dtype)
    Px = np.empty(nnz, dtype=A.dtype)
    
    # Fill in weights of P
    row_coarse = np.zeros((n, ))
    row_strong = np.zeros((n, ))
    row_coarse_sums = np.zeros((n, ))

    nnz = 0
    Pp[0] = 0
    for i in range(n):
        if splitting[i] == 1:
            Pj[nnz] = i
            Px[nnz] = 1.0
            nnz += 1
        else:          
            ctr = S.indptr[i]
            weak_sum = 0
            diag = 0
            sign = 1
            for jj in range(A.indptr[i], A.indptr[i+1]):
                col = A.indices[jj]
                if (col == i):
                    diag = A.data[jj]
                    if ctr < S.indptr[i+1] and S.indices[ctr] == col:
                        ctr += 1
                elif ctr < S.indptr[i+1] and S.indices[ctr] == col:
                    if splitting[col] == 1:
                        row_coarse[col] = 1
                    else:
                        row_strong[col] = A.data[jj]
                    ctr += 1
                else:
                    weak_sum += A.data[jj]
            
            if (diag < 0):
                sign = -1
                
            for jj in range(S.indptr[i], S.indptr[i+1]):
                col = S.indices[jj]
                if (col == i): 
                    continue
                if splitting[col] == 0:
                    for kk in range(A.indptr[col], A.indptr[col+1]):
                        col_k = A.indices[kk]
                        if col_k == col:
                            continue;
                        val_k = A.data[kk]
                        if row_coarse[col_k] and val_k * sign < 0:
                            row_coarse_sums[col] += val_k
                    if row_coarse_sums[col] == 0:
                        weak_sum += S.data[jj]
            weak_sum += diag
            
            for jj in range(S.indptr[i], S.indptr[i+1]):
                col = S.indices[jj]
                if (col == i):
                    continue
                if splitting[col] == 1:
                    weight = -S.data[jj]
                    for kk in range(AT.indptr[col], AT.indptr[col+1]):
                        row = AT.indices[kk]
                        if row == col:
                            continue
                        val = AT.data[kk]
                        if (row_coarse_sums[row] and val * sign < 0):
                            weight -= ((row_strong[row] * val) / row_coarse_sums[row])
                    weight /= weak_sum
                    Pj[nnz] = col
                    Px[nnz] = weight
                    nnz += 1
            for jj in range(S.indptr[i], S.indptr[i+1]):
                col = S.indices[jj]
                row_coarse[col] = 0
                row_strong[col] = 0.0
                row_coarse_sums[col] = 0.0
                     
        Pp[i+1] = nnz
    
    col_sum = 0
    map = np.zeros((n, 1))
    for i in range(0, n):
        map[i] = col_sum
        col_sum += splitting[i]
    for i in range(0, nnz):
        Pj[i] = map[Pj[i]]
    
    return csr_matrix((Px, Pj, Pp))
