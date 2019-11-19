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

TEST(ParBVectorRectangularTAPSpMVMedTest, TestsInUtil)
{
    setenv("PPN", "4", 1);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int vecs_in_block = 3;
    int global_rows = 24;
    int global_cols = 10;
    int local_rows, local_cols, first_local_row, first_local_col;
   
    if (rank == 0)
    {
        local_rows = 2;
        local_cols = 0; 
    
        first_local_row = 0;
        first_local_col = 0;
    }
    else if (rank == 1)
    {
        local_rows = 3;
        local_cols = 1; 
        
        first_local_row = 2;
        first_local_col = 0;
    }
    else if (rank == 2)
    {
        local_rows = 3;
        local_cols = 2;

        first_local_row = 5;
        first_local_col = 1;
    }
    else if (rank == 3)
    {
        local_rows = 1;
        local_cols = 0;

        first_local_row = 8;
        first_local_col = 0;
    }
    else if (rank == 4)
    {
        local_rows = 2;
        local_cols = 1;

        first_local_row = 9;
        first_local_col = 3;
    }
    else if (rank == 5)
    {
        local_rows = 1;
        local_cols = 1;
        
        first_local_row = 11;
        first_local_col = 4;
    }
    else if (rank == 6)
    {
        local_rows = 2;
        local_cols = 2;
        
        first_local_row = 12;
        first_local_col = 5;
    }
    else if (rank == 7)
    {
        local_rows = 2;
        local_cols = 1;
        
        first_local_row = 14;
        first_local_col = 7;
    }
    else if (rank == 8)
    {
        local_rows = 3;
        local_cols = 0;
        
        first_local_row = 16;
        first_local_col = 0;
    }
    else if (rank == 9)
    {
        local_rows = 2;
        local_cols = 1;
        
        first_local_row = 19;
        first_local_col = 8;
    }
    else if (rank == 10)
    {
        local_rows = 1;
        local_cols = 1;
        
        first_local_row = 21;
        first_local_col = 9;
    }
    else if (rank == 11)
    {
        local_rows = 3;
        local_cols = 0;
        
        first_local_row = 22;
        first_local_col = 0;
    }

    ParCSRMatrix* P = new ParCSRMatrix(global_rows, global_cols,
                                local_rows, local_cols,
                                first_local_row, first_local_col);

    if (rank == 0)
    {
        P->add_value(0, 3, 1.0);
        P->add_value(0, 8, 1.0);
        P->add_value(0, 9, 1.0);
        P->add_value(1, 4, 1.0);
        P->add_value(1, 9, 1.0);
    }
    else if (rank == 1)
    {
        P->add_value(0, 6, 1.0);
        P->add_value(0, 9, 1.0);
        P->add_value(1, 0, 1.0);
        P->add_value(1, 2, 1.0);
        P->add_value(2, 3, 1.0);
    }
    else if (rank == 2)
    {
        P->add_value(0, 1, 1.0);
        P->add_value(0, 7, 1.0);
        P->add_value(0, 8, 1.0);
        P->add_value(1, 2, 1.0);
        P->add_value(1, 5, 1.0);
        P->add_value(2, 0, 1.0);
        P->add_value(2, 9, 1.0);
    }
    else if (rank == 3)
    {
        P->add_value(0, 6, 1.0);
    }
    else if (rank == 4)
    {
        P->add_value(0, 3, 1.0);
        P->add_value(0, 5, 1.0);
        P->add_value(0, 9, 1.0);
        P->add_value(1, 0, 1.0);
        P->add_value(1, 6, 1.0);
    }
    else if (rank == 5)
    {
        P->add_value(0, 3, 1.0);
        P->add_value(0, 4, 1.0);
    }
    else if (rank == 6)
    {
        P->add_value(0, 1, 1.0);
        P->add_value(0, 5, 1.0);
        P->add_value(0, 8, 1.0);
        P->add_value(1, 2, 1.0);
        P->add_value(1, 5, 1.0);
        P->add_value(1, 6, 1.0);
        P->add_value(1, 8, 1.0);
    }
    else if (rank == 7)
    {
        P->add_value(0, 1, 1.0);
        P->add_value(0, 3, 1.0);
        P->add_value(1, 0, 1.0);
        P->add_value(1, 3, 1.0);
        P->add_value(1, 7, 1.0);
    }
    else if (rank == 8)
    {
        P->add_value(0, 1, 1.0);
        P->add_value(1, 5, 1.0);
        P->add_value(2, 0, 1.0);
    }
    else if (rank == 9)
    {
        P->add_value(0, 3, 1.0);
        P->add_value(0, 8, 1.0);
        P->add_value(1, 6, 1.0);
        P->add_value(1, 8, 1.0);
        P->add_value(1, 9, 1.0);
    }
    else if (rank == 10)
    {
        P->add_value(0, 9, 1.0);
    }
    else if (rank == 11)
    {
        P->add_value(0, 3, 1.0);
        P->add_value(0, 7, 1.0);
        P->add_value(1, 5, 1.0);
        P->add_value(2, 0, 1.0);
        P->add_value(2, 3, 1.0);
    }

    P->finalize();
    
    // Setup ParVectors //
    ParBVector x(global_cols, P->on_proc_num_cols, vecs_in_block);
    ParBVector b(global_rows, P->local_num_rows, vecs_in_block);
    ParBVector res(global_rows, P->local_num_rows, vecs_in_block);
    
    ParVector x1(global_cols, P->on_proc_num_cols);
    ParVector x2(global_cols, P->on_proc_num_cols);
    ParVector x3(global_cols, P->on_proc_num_cols);

    ParVector b1(global_rows, P->local_num_rows);
    ParVector b2(global_rows, P->local_num_rows);
    ParVector b3(global_rows, P->local_num_rows);
    
    ParVector res1(global_rows, P->local_num_rows);
    ParVector res2(global_rows, P->local_num_rows);
    ParVector res3(global_rows, P->local_num_rows);

    // Set values in ParVectors //
    for (int i = 0; i < x1.local_n; i++)
    {
        x.local->values[i] = rank;
        x.local->values[x.local_n + i] = rank*2;
        x.local->values[2*x.local_n + i] = rank*3;
        
        x1.local->values[i] = rank;
        x2.local->values[i] = rank*2;
        x3.local->values[i] = rank*3;
    }

    // Setup 3-step TAPComm for tests //
    P->tap_comm = new TAPComm(P->partition, P->off_proc_column_map, P->on_proc_column_map);
    
    // Test tap_mult
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
    
    // Test tap_mult_append
    P->tap_mult_append(x1, b1);
    P->tap_mult_append(x2, b2);
    P->tap_mult_append(x3, b3);

    P->tap_mult_append(x, b);
    
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
    
    // Test tap_residual 
    P->tap_residual(x1, b1, res1);
    P->tap_residual(x2, b2, res2);
    P->tap_residual(x3, b3, res3);
    
    P->tap_residual(x, b, res);
    
    // Test each vector in block vector
    for (int i = 0; i < res.local_n; i++)
    {
        ASSERT_NEAR(res.local->values[i], res1.local->values[i], 1e-06);
    }
    for (int i = 0; i < res.local_n; i++)
    {
        ASSERT_NEAR(res.local->values[res.local_n + i], res2.local->values[i], 1e-06);
    }
    for (int i = 0; i < res.local_n; i++)
    {
        ASSERT_NEAR(res.local->values[2*res.local_n + i], res3.local->values[i], 1e-06);
    }
    
    // Vectors to test against
    for (int i = 0; i < b.local_n; i++)
    {
        b.local->values[i] = P->partition->first_local_row + i;
        b.local->values[b.local_n + i] = (P->partition->first_local_row + i)*2;
        b.local->values[2*b.local_n + i] = (P->partition->first_local_row + i)*3;
        
        b1.local->values[i] = P->partition->first_local_row + i;
        b2.local->values[i] = (P->partition->first_local_row + i)*2;
        b3.local->values[i] = (P->partition->first_local_row + i)*3;
    }

    P->tap_mult_T(b1, x1);
    P->tap_mult_T(b2, x2);
    P->tap_mult_T(b3, x3);

    P->tap_mult_T(b, x);
    
    for (int i = 0; i < x.local_n; i++)
    {
        ASSERT_NEAR(x.local->values[i], x1.local->values[i], 1e-06);
    }
    for (int i = 0; i < x.local_n; i++)
    {
        ASSERT_NEAR(x.local->values[x.local_n + i], x2.local->values[i], 1e-06);
    }
    for (int i = 0; i < x.local_n; i++)
    {
        ASSERT_NEAR(x.local->values[2*x.local_n + i], x3.local->values[i], 1e-06);
    }
   
    MPI_Barrier(MPI_COMM_WORLD);
    // Setup 2-step TAPComm for tests //
    delete P->tap_comm;
    P->tap_comm = new TAPComm(P->partition, P->off_proc_column_map, P->on_proc_column_map, false);
    
    // Test tap_mult
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
    
    // Test tap_mult_append
    P->tap_mult_append(x1, b1);
    P->tap_mult_append(x2, b2);
    P->tap_mult_append(x3, b3);

    P->tap_mult_append(x, b);
    
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
    
    // Test tap_residual 
    P->tap_residual(x1, b1, res1);
    P->tap_residual(x2, b2, res2);
    P->tap_residual(x3, b3, res3);
    
    P->tap_residual(x, b, res);
    
    // Test each vector in block vector
    for (int i = 0; i < res.local_n; i++)
    {
        ASSERT_NEAR(res.local->values[i], res1.local->values[i], 1e-06);
    }
    for (int i = 0; i < res.local_n; i++)
    {
        ASSERT_NEAR(res.local->values[res.local_n + i], res2.local->values[i], 1e-06);
    }
    for (int i = 0; i < res.local_n; i++)
    {
        ASSERT_NEAR(res.local->values[2*res.local_n + i], res3.local->values[i], 1e-06);
    }
    
    // Vectors to test against
    for (int i = 0; i < b.local_n; i++)
    {
        b.local->values[i] = P->partition->first_local_row + i;
        b.local->values[b.local_n + i] = (P->partition->first_local_row + i)*2;
        b.local->values[2*b.local_n + i] = (P->partition->first_local_row + i)*3;
        
        b1.local->values[i] = P->partition->first_local_row + i;
        b2.local->values[i] = (P->partition->first_local_row + i)*2;
        b3.local->values[i] = (P->partition->first_local_row + i)*3;
    }

    P->tap_mult_T(b1, x1);
    P->tap_mult_T(b2, x2);
    P->tap_mult_T(b3, x3);

    P->tap_mult_T(b, x);

    for (int i = 0; i < x.local_n; i++)
    {
        ASSERT_NEAR(x.local->values[i], x1.local->values[i], 1e-06);
    }
    for (int i = 0; i < x.local_n; i++)
    {
        ASSERT_NEAR(x.local->values[x.local_n + i], x2.local->values[i], 1e-06);
    }
    for (int i = 0; i < x.local_n; i++)
    {
        ASSERT_NEAR(x.local->values[2*x.local_n + i], x3.local->values[i], 1e-06);
    }

    setenv("PPN", "16", 1);

    delete P;

} // end of TEST(ParBVectorRectangularTAPSpMVMedTest, TestsInUtil) */
