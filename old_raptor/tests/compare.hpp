// EXPECT_EQ and ASSERT_EQ are macros
// EXPECT_EQ test execution and continues even if there is a failure
// ASSERT_EQ test execution and aborts if there is a failure
// The ASSERT_* variants abort the program execution if an assertion fails 
// while EXPECT_* variants continue with the run.

#include "gtest/gtest.h"

#ifndef RAPTOR_TEST_COMPARE_HPP
#define RAPTOR_TEST_COMPARE_HPP

#include "core/types.hpp"
#include "core/matrix.hpp"

void compare(CSRMatrix* A, CSRMatrix* A_rap)
{
    int start, end;

    A->sort();
    A_rap->sort();

    if (A->n_rows == A->n_cols)
    {
        A->move_diag();
        A_rap->move_diag();
    }

    ASSERT_EQ(A->n_rows, A_rap->n_rows);
    ASSERT_EQ(A->n_cols, A_rap->n_cols);
    ASSERT_EQ(A->nnz, A_rap->nnz);
    ASSERT_EQ(A->idx1[0], A_rap->idx1[0]);
    for (int i = 0; i < A->n_rows; i++)
    {
        ASSERT_EQ(A->idx1[i+1], A_rap->idx1[i+1]);
        start = A->idx1[i];
        end = A->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            ASSERT_EQ(A->idx2[j], A_rap->idx2[j]);
            ASSERT_NEAR(A->vals[j], A_rap->vals[j], 1e-06);
        }
    }
}

void compare_pattern(CSRMatrix* A, CSRMatrix* A_rap)
{
    int start, end;

    A->sort();
    A_rap->sort();
    A->move_diag();
    A_rap->move_diag();

    ASSERT_EQ(A->n_rows, A_rap->n_rows);
    ASSERT_EQ(A->n_cols, A_rap->n_cols);
    ASSERT_EQ(A->nnz, A_rap->nnz);
    ASSERT_EQ(A->idx1[0], A_rap->idx1[0]);
    for (int i = 0; i < A->n_rows; i++)
    {
        ASSERT_EQ(A->idx1[i+1], A_rap->idx1[i+1]);
        start = A->idx1[i];
        end = A->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            ASSERT_EQ(A->idx2[j], A_rap->idx2[j]);
        }
    }
}

#endif
