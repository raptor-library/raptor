// EXPECT_EQ and ASSERT_EQ are macros
// EXPECT_EQ test execution and continues even if there is a failure
// ASSERT_EQ test execution and aborts if there is a failure
// The ASSERT_* variants abort the program execution if an assertion fails 
// while EXPECT_* variants continue with the run.

#include "gtest/gtest.h"

#ifndef RAPTOR_TEST_PAR_COMPARE_HPP
#define RAPTOR_TEST_PAR_COMPARE_HPP
#include "raptor/core/types.hpp"
#include "raptor/core/par_matrix.hpp"

namespace raptor {
void compare(ParCSRMatrix* A, ParCSRMatrix* A_rap)
{
    int start, end;

    ASSERT_EQ(A->global_num_rows, A_rap->global_num_rows);
    ASSERT_EQ(A->global_num_cols, A_rap->global_num_cols);
    ASSERT_EQ(A->local_num_rows, A_rap->local_num_rows);
    ASSERT_EQ(A->on_proc_num_cols, A_rap->on_proc_num_cols);
    ASSERT_EQ(A->off_proc_num_cols, A_rap->off_proc_num_cols);

    A->on_proc->sort();
    A->on_proc->move_diag();
    A->off_proc->sort();
    A_rap->on_proc->sort();
    A_rap->on_proc->move_diag();
    A_rap->off_proc->sort();

    ASSERT_EQ(A->on_proc->idx1[0], A_rap->on_proc->idx1[0]);
    ASSERT_EQ(A->off_proc->idx1[0], A_rap->off_proc->idx1[0]);
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_EQ(A->on_proc->idx1[i+1], A_rap->on_proc->idx1[i+1]);
        start = A->on_proc->idx1[i];
        end = A->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            ASSERT_EQ(A->on_proc->idx2[j], A_rap->on_proc->idx2[j]);
            ASSERT_NEAR(A->on_proc->vals[j], A_rap->on_proc->vals[j], 1e-06);
        }

        ASSERT_EQ(A->off_proc->idx1[i+1], A_rap->off_proc->idx1[i+1]);
        start = A->off_proc->idx1[i];
        end = A->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            ASSERT_EQ(A->off_proc->idx2[j], A_rap->off_proc->idx2[j]);
            ASSERT_NEAR(A->off_proc->vals[j], A_rap->off_proc->vals[j], 1e-06);
        }
    }
}

void compare_pattern(ParCSRMatrix* A, ParCSRMatrix* A_rap)
{
    int start, end;

    ASSERT_EQ(A->global_num_rows, A_rap->global_num_rows);
    ASSERT_EQ(A->global_num_cols, A_rap->global_num_cols);
    ASSERT_EQ(A->local_num_rows, A_rap->local_num_rows);
    ASSERT_EQ(A->on_proc_num_cols, A_rap->on_proc_num_cols);
    ASSERT_EQ(A->off_proc_num_cols, A_rap->off_proc_num_cols);

    A->on_proc->sort();
    A->on_proc->move_diag();
    A->off_proc->sort();
    A_rap->on_proc->sort();
    A_rap->on_proc->move_diag();
    A_rap->off_proc->sort();

    ASSERT_EQ(A->on_proc->idx1[0], A_rap->on_proc->idx1[0]);
    ASSERT_EQ(A->off_proc->idx1[0], A_rap->off_proc->idx1[0]);
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_EQ(A->on_proc->idx1[i+1], A_rap->on_proc->idx1[i+1]);
        start = A->on_proc->idx1[i];
        end = A->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            ASSERT_EQ(A->on_proc->idx2[j], A_rap->on_proc->idx2[j]);
        }

        ASSERT_EQ(A->off_proc->idx1[i+1], A_rap->off_proc->idx1[i+1]);
        start = A->off_proc->idx1[i];
        end = A->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            ASSERT_EQ(A->off_proc->idx2[j], A_rap->off_proc->idx2[j]);
        }
    }
}

void remove_empty_cols(ParCSRMatrix* S)
{
    std::vector<int> col_exists(S->off_proc_num_cols, 0);
    for (std::vector<int>::iterator it = S->off_proc->idx2.begin();
            it != S->off_proc->idx2.end(); ++it)
    {
        col_exists[*it] = 1;
    }
    int ctr = 0;
    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        if (col_exists[i])
        {
            col_exists[i] = ctr;
            S->off_proc_column_map[ctr++] = S->off_proc_column_map[i];
        }
    }
    S->off_proc_column_map.resize(ctr);
    S->off_proc_num_cols = ctr;
    for (std::vector<int>::iterator it = S->off_proc->idx2.begin();
            it != S->off_proc->idx2.end(); ++it)
    {
        *it = col_exists[*it];
    }
}

}
#endif
