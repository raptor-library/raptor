#include "matmult.hpp"

index_t map_to_global(index_t i, std::vector<index_t> map)
{
    return map[i];
}

index_t map_to_global(index_t i, index_t addition)
{
    return i + addition;
}


// Dot product u^T*v (with global indices)
template <typename UType, typename VType>
data_t dot(index_t size_u, index_t* local_u, UType map_u, data_t* data_u, 
    index_t size_v, index_t* local_v, VType map_v, data_t* data_v )
{
    index_t ctr_u = 0;
    index_t ctr_v = 0;
    data_t result = 0.0;

    index_t k_u = map_to_global(local_u[ctr_u], map_u);
    index_t k_v = map_to_global(local_v[ctr_v], map_v);
    while (ctr_u < size_u && ctr_v < size_v)
    {
        if (k_u == k_v)
        {
            result += data_u[ctr_u++] * data_v[ctr_v++];
            if (ctr_u < size_u && ctr_v < size_v)
            {
                k_u = map_to_global(local_u[ctr_u], map_u);
                k_v = map_to_global(local_v[ctr_v], map_v);
            }
        }
        else if (k_u > k_v)
        {
            ctr_v++;
            if (ctr_v < size_v)
            {
                k_v = map_to_global(local_v[ctr_v], map_v);
            }
        }
        else
        {
            ctr_u++;
            if (ctr_u < size_u)
            {
                k_u = map_to_global(local_u[ctr_u], map_u);
            }
        }
    }
    return result;
}

template <typename AType, typename BType>
data_t matmult(Matrix* A, Matrix* B, AType map_A,
        BType map_B, index_t col, index_t row,
        index_t col_start, index_t col_end)
{
    index_t row_start = A->indptr[row];
    index_t row_end = A->indptr[row+1];

    index_t size_A = row_end - row_start;
    index_t size_B = col_end - col_start;

    index_t* local_A = &(A->indices[row_start]);
    index_t* local_B = &(B->indices[col_start]);

    data_t* data_A = &(A->data[row_start]);
    data_t* data_B = &(B->data[col_start]);

    return dot<AType, BType>(size_A, local_A, map_A, data_A,
                    size_B, local_B, map_B, data_B);
}

template <typename AType, typename BType, typename CType>
void seq_mm(Matrix* A, Matrix* B, ParMatrix* C, AType map_A,
        BType map_B, CType map_C, index_t col)
{
    index_t col_start = B->indptr[col];
    index_t col_end = B->indptr[col+1];
    for (index_t row = 0; row < A->n_rows; row++)
    {
        data_t cij = matmult<AType, BType> (A, B, map_A,
             map_B, col, row, col_start, col_end);
        index_t global_col = map_to_global(col, map_C);
        C->add_value(row, global_col, cij);
    }
}

template <typename AType, typename BType, typename CType>
void seq_mm(Matrix* A, Matrix* B, ParMatrix* C, AType map_A,
        BType map_B, CType map_C)
{
    data_t cij;
    index_t global_col;

    for (index_t col = 0; col < B->n_cols; col++)
    {
        seq_mm<AType, BType, CType> (A, B, C, map_A,
            map_B, map_C, col);
    }
}

