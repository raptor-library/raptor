#ifndef RAPTOR_TEST_COMPARE_HPP
#define RAPTOR_TEST_COMPARE_HPP

#include "core/types.hpp"
#include "core/matrix.hpp"

void compare(CSRMatrix* A, CSRMatrix* A_rap)
{
    int start, end;

    A->sort();
    A_rap->sort();
    A->move_diag();
    A_rap->move_diag();

    assert(A->n_rows == A_rap->n_rows);
    assert(A->n_cols == A_rap->n_cols);
    assert(A->nnz == A_rap->nnz);
    assert(A->idx1[0] == A_rap->idx1[0]);
    for (int i = 0; i < A->n_rows; i++)
    {
        assert(A->idx1[i+1] == A_rap->idx1[i+1]);
        start = A->idx1[i];
        end = A->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            assert(A->idx2[j] == A_rap->idx2[j]);
            assert(fabs(A->vals[j] - A_rap->vals[j]) < 1e-06);
        }
    }
}

void compare(CSRMatrix* A, CSRBoolMatrix* A_rap)
{
    int start, end;

    A->sort();
    A_rap->sort();
    A->move_diag();
    A_rap->move_diag();

    assert(A->n_rows == A_rap->n_rows);
    assert(A->n_cols == A_rap->n_cols);
    assert(A->nnz == A_rap->nnz);
    assert(A->idx1[0] == A_rap->idx1[0]);
    for (int i = 0; i < A->n_rows; i++)
    {
        assert(A->idx1[i+1] == A_rap->idx1[i+1]);
        start = A->idx1[i];
        end = A->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            assert(A->idx2[j] == A_rap->idx2[j]);
        }
    }
}

#endif
