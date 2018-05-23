// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "assert.h"
#include "core/matrix.hpp"

using namespace raptor;

// TODO -- currently assumes partitions are the same 
Matrix* Matrix::add(CSRMatrix* B)
{
    return NULL;
}
Matrix* Matrix::subtract(CSRMatrix* B)
{
    return NULL;
}

CSRMatrix* CSRMatrix::add(CSRMatrix* B)
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
            C->idx2.push_back(idx2[j]);
            C->vals.push_back(vals[j]);
        }
        start = B->idx1[i];
        end = B->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            C->idx2.push_back(B->idx2[j]);
            C->vals.push_back(B->vals[j]);
        }
        C->idx1[i+1] = C->idx2.size();
    }
    C->nnz = C->idx2.size();
    C->sort();
    C->remove_duplicates();
    C->move_diag();    

    return C;
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
            C->idx2.push_back(idx2[j]);
            C->vals.push_back(vals[j]);
        }
        start = B->idx1[i];
        end = B->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            C->idx2.push_back(B->idx2[j]);
            C->vals.push_back(-B->vals[j]);
        }
        C->idx1[i+1] = C->idx2.size();
    }
    C->nnz = C->idx2.size();
    C->sort();
    C->remove_duplicates();
    C->move_diag();    

    return C;
}


