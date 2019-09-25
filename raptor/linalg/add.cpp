// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "assert.h"
#include "core/matrix.hpp"

using namespace raptor;

// TODO -- currently assumes partitions are the same 
Matrix* Matrix::add(CSRMatrix* B, bool remove_dup)
{
    CSRMatrix* A = to_CSR();
    CSRMatrix* C = new CSRMatrix(n_rows, n_cols, 2*nnz);
    A->add_append(B, C, remove_dup);
    delete A;
    return C;
}
void Matrix::add_append(CSRMatrix* B, CSRMatrix* C, bool remove_dup)
{
    CSRMatrix* A = to_CSR();
    A->add_append(B, C, remove_dup);
    delete A;
}
Matrix* Matrix::subtract(CSRMatrix* B)
{
    CSRMatrix* A = to_CSR();
    CSRMatrix* C = A->subtract(B);
    delete A;
    return C;
}


CSRMatrix* CSRMatrix::add(CSRMatrix* B, bool remove_dup)
{
    CSRMatrix* C = new CSRMatrix(n_rows, n_cols, 2*nnz);
    add_append(B, C, remove_dup);
    return C;
}

void CSRMatrix::add_append(CSRMatrix* B, CSRMatrix* C, bool remove_dup)
{
    int start, end;

    C->resize(n_rows, n_cols);
    int C_nnz = nnz + B->nnz;
    C->idx2.resize(C_nnz);
    C->vals.resize(C_nnz);

    C_nnz = 0;
    C->idx1[0] = 0;
    for (int i = 0; i < n_rows; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        std::copy(idx2.begin() + start,
                idx2.begin() + end,
                C->idx2.begin() + C_nnz);
        std::copy(vals.begin() + start,
                vals.begin() + end,
                C->vals.begin() + C_nnz);
        C_nnz += (end - start);

        start = B->idx1[i];
        end = B->idx1[i+1];
        std::copy(B->idx2.begin() + start,
                B->idx2.begin() + end,
                C->idx2.begin() + C_nnz);
        std::copy(B->vals.begin() + start,
                B->vals.begin() + end,
                C->vals.begin() + C_nnz);
        C_nnz += (end - start);

        C->idx1[i+1] = C_nnz;
    }
    C->nnz = C_nnz;
    C->sort();
    if (remove_dup) 
        C->remove_duplicates();
}

CSRMatrix* CSRMatrix::subtract(CSRMatrix* B)
{
    int start, end;

    assert(n_rows == B->n_rows);
    assert(n_cols == B->n_cols);

    CSRMatrix* C = new CSRMatrix(n_rows, n_cols, 2*nnz);
    C->idx1[0] = 0;
    for (int i = 0; i < n_rows; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            C->idx2.emplace_back(idx2[j]);
            C->vals.emplace_back(vals[j]);
        }
        start = B->idx1[i];
        end = B->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            C->idx2.emplace_back(B->idx2[j]);
            C->vals.emplace_back(-B->vals[j]);
        }
        C->idx1[i+1] = C->idx2.size();
    }
    C->nnz = C->idx2.size();
    C->sort();
    C->remove_duplicates();

    return C;
}


