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
    setenv("PPN", "4", 1);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    FILE* f;
    double b_val;
    int vecs_in_block = 3;
    int grid[2] = {25, 25};
    double eps = 0.001;
    double theta = M_PI/8.0;
    double* stencil = diffusion_stencil_2d(eps, theta);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);
    
    double strong_threshold = 0.0;

    ParMultilevel* ml;
    ml = new ParRugeStubenSolver(strong_threshold, CLJP, ModClassical, Classical, SOR);
    ml->setup(A);
    ml->print_hierarchy();

    ParCSRMatrix* P = ml->levels[0]->P;
    ParVector x1(ml->levels[2]->x);
    ParVector x2(ml->levels[2]->x);
    ParVector x3(ml->levels[2]->x);
    ParVector b1(ml->levels[1]->x);
    ParVector b2(ml->levels[1]->x);
    ParVector b3(ml->levels[1]->x);

    ParBVector x(x1.global_n, x1.local_n, vecs_in_block);
    ParBVector b(b1.global_n, b1.local_n, vecs_in_block);
    ParBVector res(b1.global_n, b1.local_n, vecs_in_block);
    
    ParVector res1(b1.global_n, b1.local_n);
    ParVector res2(b2.global_n, b2.local_n);
    ParVector res3(b3.global_n, b3.local_n);

    P->tap_comm = new TAPComm(P->partition, P->off_proc_column_map, P->on_proc_column_map);

    // Set vectors for testing
    b.set_const_value(1.0);

    x1.set_rand_values(rank + 1);
    x2.set_rand_values(rank + 2);
    x3.set_rand_values(rank + 3);
    b1.set_const_value(1.0);
    b2.set_const_value(2.0);
    b3.set_const_value(3.0);

    // Set block vector x for testing against
    for (int i = 0; i < x.local_n; i++)
    {
        x.local->values[i] = x1.local->values[i];
        x.local->values[x.local_n + i] = x2.local->values[i];
        x.local->values[2*x.local_n + i] = x3.local->values[i];
    }

    /*for (int i = 0; i < b.local_n; i++)
    {
        b.local->values[b.local_n + i] = 2.0;
        b.local->values[2*b.local_n + i] = 3.0;
    }
    
    // Res vectors to test against
    P->tap_residual(x1, b1, res1);
    P->tap_residual(x2, b2, res2);
    P->tap_residual(x3, b3, res3);

    P->tap_residual(x, b, res);
    
    // Test each vector in block vector
    for (int i = 0; i < P->local_num_rows; i++)
    {
        ASSERT_NEAR(res.local->values[i], res1.local->values[i], 1e-06);
    }
    for (int i = 0; i < P->local_num_rows; i++)
    {
        ASSERT_NEAR(res.local->values[P->local_num_rows + i], res2.local->values[i], 1e-06);
    }
    for (int i = 0; i < P->local_num_rows; i++)
    {
        ASSERT_NEAR(res.local->values[2*P->local_num_rows + i], res3.local->values[i], 1e-06);
    }*/
    
    // Vectors to test against
    P->tap_mult(x1, b1);
    P->tap_mult(x2, b2);
    P->tap_mult(x3, b3);
    
    P->tap_mult(x, b);

    // Test each vector in block vector
    for (int i = 0; i < b.local_n; i++)
    {
        ASSERT_NEAR(b.local->values[i], b1.local->values[i], 1e-06);
    }
    for (int i = 0; i < b.local_n; i++)
    {
        ASSERT_NEAR(b.local->values[b.local_n + i], b2.local->values[i], 1e-06);
    }
    for (int i = 0; i < b.local_n; i++)
    {
        ASSERT_NEAR(b.local->values[2*b.local_n + i], b3.local->values[i], 1e-06);
    }
    
    // Vectors to test against
    /*A->tap_mult_T(x1, b1);
    A->tap_mult_T(x2, b2);
    A->tap_mult_T(x3, b3);

    A->tap_mult_T(x, b);

    // Test each vector in block vector
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(b.local->values[i], b1.local->values[i], 1e-06);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(b.local->values[A->local_num_rows + i], b2.local->values[i], 1e-06);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(b.local->values[2*A->local_num_rows + i], b3.local->values[i], 1e-06);
    }*/
    
    // Vectors to test against
    /*A->tap_mult_append(x1, b1);
    A->tap_mult_append(x2, b2);
    A->tap_mult_append(x3, b3);

    A->tap_mult_append(x, b);

    // Test each vector in block vector
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(x.local->values[i], x1.local->values[i], 1e-06);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(x.local->values[A->local_num_rows + i], x2.local->values[i], 1e-06);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(x.local->values[2*A->local_num_rows + i], x3.local->values[i], 1e-06);
    }*/
    
    // **************** SAME TESTS WITH SIMPLE TAP COMM **************** //
    /*delete A->tap_comm;
    A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map, A->on_proc_column_map, false);
   
    // Reset b's 
    b1.set_const_value(1.0);
    b2.set_const_value(2.0);
    b3.set_const_value(3.0);
    
    // Res vectors to test against
    A->tap_residual(x1, b1, res1);
    A->tap_residual(x2, b2, res2);
    A->tap_residual(x3, b3, res3);
    
    // Set block vector x for testing against
    b.set_const_value(1.0);
    for (int i = 0; i < A->local_num_rows; i++)
    {
        b.local->values[A->local_num_rows + i] = 2.0;
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        b.local->values[2*A->local_num_rows + i] = 3.0;
    }

    A->tap_residual(x, b, res);
    
    // Test each vector in block vector
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(res.local->values[i], res1.local->values[i], 1e-06);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(res.local->values[A->local_num_rows + i], res2.local->values[i], 1e-06);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(res.local->values[2*A->local_num_rows + i], res3.local->values[i], 1e-06);
    }
    
    // Vectors to test against
    A->tap_mult(x1, b1);
    A->tap_mult(x2, b2);
    A->tap_mult(x3, b3);

    A->tap_mult(x, b);

    // Test each vector in block vector
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(b.local->values[i], b1.local->values[i], 1e-06);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(b.local->values[A->local_num_rows + i], b2.local->values[i], 1e-06);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(b.local->values[2*A->local_num_rows + i], b3.local->values[i], 1e-06);
    }

    // Vectors to test against
    A->tap_mult_T(x1, b1);
    A->tap_mult_T(x2, b2);
    A->tap_mult_T(x3, b3);

    A->tap_mult_T(x, b);

    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(b.local->values[i], b1.local->values[i], 1e-06);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(b.local->values[A->local_num_rows + i], b2.local->values[i], 1e-06);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(b.local->values[2*A->local_num_rows + i], b3.local->values[i], 1e-06);
    }
    
    // Vectors to test against
    A->tap_mult_append(x1, b1);
    A->tap_mult_append(x2, b2);
    A->tap_mult_append(x3, b3);

    A->tap_mult_append(x, b);

    // Test each vector in block vector
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(x.local->values[i], x1.local->values[i], 1e-06);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(x.local->values[A->local_num_rows + i], x2.local->values[i], 1e-06);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(x.local->values[2*A->local_num_rows + i], x3.local->values[i], 1e-06);
    }*/

    delete A;
    delete ml;
    delete[] stencil;

    setenv("PPN", "16", 1);

} // end of TEST(ParBVectorAnisoTAPSpMVTest, TestsInUtil) //
