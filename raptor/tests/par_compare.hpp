#ifndef RAPTOR_TEST_PAR_COMPARE_HPP
#define RAPTOR_TEST_PAR_COMPARE_HPP
#include "core/types.hpp"
#include "core/par_matrix.hpp"

void compare(ParCSRMatrix* A, ParCSRMatrix* A_rap)
{
    int start, end;

    assert(A->global_num_rows == A_rap->global_num_rows);
    assert(A->global_num_cols == A_rap->global_num_cols);
    assert(A->local_num_rows == A_rap->local_num_rows);
    assert(A->on_proc_num_cols == A_rap->on_proc_num_cols);
    assert(A->off_proc_num_cols == A_rap->off_proc_num_cols);

    A->on_proc->sort();
    A->on_proc->move_diag();
    A->off_proc->sort();
    A_rap->on_proc->sort();
    A_rap->on_proc->move_diag();
    A_rap->off_proc->sort();

    assert(A->on_proc->idx1[0] == A_rap->on_proc->idx1[0]);
    assert(A->off_proc->idx1[0] == A_rap->off_proc->idx1[0]);
    for (int i = 0; i < A->local_num_rows; i++)
    {
        assert(A->on_proc->idx1[i+1] == A_rap->on_proc->idx1[i+1]);
        start = A->on_proc->idx1[i];
        end = A->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            assert(A->on_proc->idx2[j] == A_rap->on_proc->idx2[j]);
            assert(fabs(A->on_proc->vals[j] - A_rap->on_proc->vals[j]) < 1e-06);
        }

        assert(A->off_proc->idx1[i+1] == A_rap->off_proc->idx1[i+1]);
        start = A->off_proc->idx1[i];
        end = A->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            assert(A->off_proc->idx2[j] == A_rap->off_proc->idx2[j]);
            assert(fabs(A->off_proc->vals[j] - A_rap->off_proc->vals[j]) < 1e-06);
        }
    }
}

void compare(ParCSRMatrix* A, ParCSRBoolMatrix* A_rap)
{
    int start, end;

    assert(A->global_num_rows == A_rap->global_num_rows);
    assert(A->global_num_cols == A_rap->global_num_cols);
    assert(A->local_num_rows == A_rap->local_num_rows);
    assert(A->on_proc_num_cols == A_rap->on_proc_num_cols);
    assert(A->off_proc_num_cols == A_rap->off_proc_num_cols);

    A->on_proc->sort();
    A->on_proc->move_diag();
    A->off_proc->sort();
    A_rap->on_proc->sort();
    A_rap->on_proc->move_diag();
    A_rap->off_proc->sort();

    assert(A->on_proc->idx1[0] == A_rap->on_proc->idx1[0]);
    assert(A->off_proc->idx1[0] == A_rap->off_proc->idx1[0]);
    for (int i = 0; i < A->local_num_rows; i++)
    {
        assert(A->on_proc->idx1[i+1] == A_rap->on_proc->idx1[i+1]);
        start = A->on_proc->idx1[i];
        end = A->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            assert(A->on_proc->idx2[j] == A_rap->on_proc->idx2[j]);
        }

        assert(A->off_proc->idx1[i+1] == A_rap->off_proc->idx1[i+1]);
        start = A->off_proc->idx1[i];
        end = A->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            assert(A->off_proc->idx2[j] == A_rap->off_proc->idx2[j]);
        }
    }
}


#endif
