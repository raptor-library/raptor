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
    ParBVector *tmp = new ParBVector(A->global_num_rows, A->local_num_rows, vecs_in_block);

    // Vectors to test against
    ParVector tmp_single(A->global_num_cols, A->on_proc_num_cols);

    ParVector x1(A->global_num_cols, A->on_proc_num_cols);
    ParVector x2(A->global_num_cols, A->on_proc_num_cols);
    ParVector x3(A->global_num_cols, A->on_proc_num_cols);

    ParVector b1(A->global_num_cols, A->on_proc_num_cols);
    ParVector b2(A->global_num_cols, A->on_proc_num_cols);
    ParVector b3(A->global_num_cols, A->on_proc_num_cols);
   
    // Set rand values 
    x1.set_rand_values();
    x2.set_rand_values();
    x3.set_rand_values();

    for (int i = 0; i < A->local_num_rows; i++)
    {
        x->local->values[i] = x1.local->values[i];
        x->local->values[A->local_num_rows + i] = x2.local->values[i];
        x->local->values[2*A->local_num_rows + i] = x3.local->values[i];
    }

    A->mult(x1, b1);
    A->mult(x2, b2);
    A->mult(x3, b3);

    A->mult(*x, *b);
    
    // Test Jacobi
    jacobi(A, x1, b1, tmp_single);
    jacobi(A, x2, b2, tmp_single);
    jacobi(A, x3, b3, tmp_single);
    jacobi(A, *x, *b, *tmp);

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
    
    // Test SOR
    sor(A, x1, b1, tmp_single);
    sor(A, x2, b2, tmp_single);
    sor(A, x3, b3, tmp_single);
    sor(A, *x, *b, *tmp);
    
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

    // Test SSOR
    ssor(A, x1, b1, tmp_single);
    ssor(A, x2, b2, tmp_single);
    ssor(A, x3, b3, tmp_single);
    ssor(A, *x, *b, *tmp);

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

    delete x;
    delete b;
    delete tmp;
    delete A;
    delete[] stencil;

} // end of TEST(ParBVectorAnisoTAPSpMVTest, TestsInUtil) //
