// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/par_stencil.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp = RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //

TEST(TAPAnisoBVMultTTest, TestsInUtil)
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

    setenv("PPN", "4", 1);

    ParBVector x(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col, vecs_in_block);
    ParBVector b(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col, vecs_in_block);
    
    ParVector x1(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col);
    ParVector b1(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col);

    ParVector x2(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col);
    ParVector b2(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col);
    
    ParVector x3(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col);
    ParVector b3(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col);

    x1.set_rand_values();
    x2.set_rand_values();
    x3.set_rand_values();

    // vectors to test against    
    A->tap_mult_T(x1, b1);
    A->tap_mult_T(x2, b2);
    A->tap_mult_T(x3, b3);

    // Set block vector x for testing against
    for (int i=0; i<A->local_num_rows; i++)
    {
        x.local->values[i] = x1.local->values[i];
    }
    for (int i=0; i<A->local_num_rows; i++)
    {
        x.local->values[A->local_num_rows + i] = x2.local->values[i];
    }
    for (int i=0; i<A->local_num_rows; i++)
    {
        x.local->values[2*A->local_num_rows + i] = x3.local->values[i];
    }

    A->tap_mult_T(x, b);
    
    // Test first vector in block 
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(b.local->values[i], b1.local->values[i], 1e-06);
    }
    
    // Test second vector in block
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(b.local->values[A->local_num_rows + i], b2.local->values[i], 1e-06);
    }
    
    // Test third vector in block
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(b.local->values[2*A->local_num_rows + i], b3.local->values[i], 1e-06);
    }

    // TEST SAME WITH SIMPLE TAP COMM
    delete A->tap_comm;
    A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map, A->on_proc_column_map, false);
    
    // res vectors to test against    
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
    
    delete A;
    delete[] stencil;

    setenv("PPN", "16", 1);

} // end of TEST(ParAnisoBVMultTTest, TestsInUtil) //
