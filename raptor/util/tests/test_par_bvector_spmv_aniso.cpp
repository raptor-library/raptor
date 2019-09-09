// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "gtest/gtest.h"
#include "raptor.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp = RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //

TEST(ParBVectorAnisoTAPSpMVTest, TestsInUtil)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double b_val;
    int vecs_in_block = 3;
    int grid[2] = {25, 25};
    double eps = 0.001;
    double theta = M_PI/8.0;
    double* stencil = diffusion_stencil_2d(eps, theta);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);

    ParBVector *x = new ParBVector(A->global_num_cols, A->on_proc_num_cols, vecs_in_block);
    ParBVector *b = new ParBVector(A->global_num_rows, A->local_num_rows, vecs_in_block);

    // Vectors to test against
    ParVector x1(A->global_num_cols, A->on_proc_num_cols);
    ParVector b1(A->global_num_cols, A->on_proc_num_cols);

    ParVector x2(A->global_num_cols, A->on_proc_num_cols);
    ParVector b2(A->global_num_cols, A->on_proc_num_cols);

    ParVector x3(A->global_num_cols, A->on_proc_num_cols);
    ParVector b3(A->global_num_cols, A->on_proc_num_cols);

    x1.set_const_value(1.0);
    x2.set_const_value(2.0);
    x3.set_const_value(3.0);

    x->set_const_value(1.0);
    std::vector<double> alphas = {1.0, 2.0, 3.0};
    x->scale(1.0, &(alphas[0]));

    A->mult(x1, b1);
    A->mult(x2, b2);
    A->mult(x3, b3);

    A->mult(*x, *b);

    // Test each vector in block vector
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(b->local->values[i], b1.local->values[i], 1e-06);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(b->local->values[A->local_num_rows + i], b2.local->values[i], 1e-06);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(b->local->values[2*A->local_num_rows + i], b3.local->values[i], 1e-06);
    }
    
    b1.set_const_value(1.0);
    b2.set_const_value(2.0);
    b3.set_const_value(3.0);

    b->set_const_value(1.0);
    b->scale(1.0, &(alphas[0]));
    
    A->mult_T(b1, x1);
    A->mult_T(b2, x2);
    A->mult_T(b3, x3);

    A->mult_T(*b, *x);
    
    // Test each vector in block vector
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(x->local->values[i], x1.local->values[i], 1e-06);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(x->local->values[A->local_num_rows + i], x2.local->values[i], 1e-06);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(x->local->values[2*A->local_num_rows + i], x3.local->values[i], 1e-06);
    }

    for (int i = 0; i < A->on_proc_num_cols; i++)
    {
        x1.local->values[i] = A->partition->first_local_col + i;
        for (int v = 0; v < vecs_in_block; v++)
        {
            x->local->values[i + v*x->local_n] = A->partition->first_local_col + i;
        }
    }
    A->mult(x1, b1);
    A->mult(*x, *b);
    for (int i = 0; i < A->local_num_rows; i++)
    {
        for (int v = 0; v < vecs_in_block; v++)
        {
            ASSERT_NEAR(b->local->values[i + v*b->local_n], b1.local->values[i], 1e-06);
        }
    }

    for (int i = 0; i < A->local_num_rows; i++)
    {
        b1.local->values[i] = A->partition->first_local_col + i;
        for (int v = 0; v < vecs_in_block; v++)
        {
            b->local->values[i + v*b->local_n] = A->partition->first_local_row + i;
        }
    }
    A->mult_T(b1, x1);
    A->mult_T(*b, *x);
    for (int i = 0; i < A->on_proc_num_cols; i++)
    {
        for (int v = 0; v < vecs_in_block; v++)
        {
            ASSERT_NEAR(x->local->values[i + v*x->local_n], x1.local->values[i], 1e-06);
        }
    }

    delete x;
    delete b;
    delete A;
    delete[] stencil;

} // end of TEST(ParBVectorAnisoTAPSpMVTest, TestsInUtil) //
