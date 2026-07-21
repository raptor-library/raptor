// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <iostream>
#include <fstream>

#include "gtest/gtest.h"

#include "raptor/raptor.hpp"
#include "raptor/tests/par_compare.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp = RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //

TEST(TestParProlongation, TestsInAggregation)
{ 
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::vector<int> states;
    std::vector<int> off_proc_states;

    ParCSRMatrix* A;
    ParCSRMatrix* T;

    const char* A0_fn = "../../../../test_data/sas_A0.pm";
    const char* T0_fn = "../../../../test_data/sas_T0.pm";
    const char* P0_fn = "../../../../test_data/sas_P0.pm";

    A = readParMatrix(A0_fn);
    T = readParMatrix(T0_fn);

    ParCSRMatrix* P_py = readParMatrix(P0_fn);
    ParCSRMatrix* P = jacobi_prolongation(A, T);

    compare(P, P_py);

    delete P;
    delete P_py;
    delete A;
    delete T;

} // end of TEST(TestParSplitting, TestsInRuge_Stuben) //

TEST(TestParProlongation, JacobiBSRLocal)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    if (num_procs < 2)
    {
        GTEST_SKIP() << "at least two MPI ranks required";
    }

    // each rank owns one 2x2 block row
    const int block_size = 2;
    const int global_num_rows = block_size*num_procs;
    const int first_row = block_size*rank;
    const int next_rank = (rank + 1) % num_procs;
    const int next_row = block_size*next_rank;

    ParCSRMatrix* A_csr = new ParCSRMatrix(global_num_rows, global_num_rows);
    // each rank owns the following
    A_csr->add_value(0, first_row, 4.0);
    A_csr->add_value(0, first_row + 1, 1.0);
    A_csr->add_value(0, next_row, -1.0);
    A_csr->add_value(1, first_row, 2.0);
    A_csr->add_value(1, first_row + 1, 3.0);
    A_csr->add_value(1, next_row + 1, -2.0);
    A_csr->on_proc->idx1[1] = 2;
    A_csr->on_proc->idx1[2] = 4;
    A_csr->off_proc->idx1[1] = 1;
    A_csr->off_proc->idx1[2] = 2;
    A_csr->finalize();

    // identity T
    ParCSRMatrix* T_csr = new ParCSRMatrix(global_num_rows, global_num_rows);
    T_csr->add_value(0, first_row, 1.0);
    T_csr->add_value(1, first_row + 1, 1.0);
    T_csr->on_proc->idx1[1] = 1;
    T_csr->on_proc->idx1[2] = 2;
    T_csr->finalize();

    ParBSRMatrix* A = A_csr->to_ParBSR(block_size, block_size);
    ParBSRMatrix* T = T_csr->to_ParBSR(block_size, block_size);
    delete A_csr;
    delete T_csr;

    ParBSRMatrix* P = jacobi_prolongation(A, T, false, 1.0, 1, prolongation_weighting::local);
    P->sort();

    BSRMatrix* P_on = static_cast<BSRMatrix*>(P->on_proc);
    BSRMatrix* P_off = static_cast<BSRMatrix*>(P->off_proc);
    ASSERT_EQ(P->local_num_rows, 1);
    ASSERT_EQ(P_on->nnz, 1);
    ASSERT_EQ(P_off->nnz, 1);

    const double* local_block = P_on->block_vals[0];
    EXPECT_NEAR(local_block[0], 1.0/3.0, 1e-12);
    EXPECT_NEAR(local_block[1], -1.0/6.0, 1e-12);
    EXPECT_NEAR(local_block[2], -2.0/7.0, 1e-12);
    EXPECT_NEAR(local_block[3], 4.0/7.0, 1e-12);

    ASSERT_EQ(P->off_proc_column_map[P_off->idx2[0]], next_rank);
    const double* remote_block = P_off->block_vals[0];
    EXPECT_NEAR(remote_block[0], 1.0/6.0, 1e-12);
    EXPECT_NEAR(remote_block[1], 0.0, 1e-12);
    EXPECT_NEAR(remote_block[2], 0.0, 1e-12);
    EXPECT_NEAR(remote_block[3], 2.0/7.0, 1e-12);

    delete P;
    delete T;
    delete A;
}

TEST(TestParProlongation, JacobiBSRBlock)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    if (num_procs < 2)
    {
        GTEST_SKIP() << "at least two MPI ranks required";
    }

    // each rank owns one 2x2 block row
    const int block_size = 2;
    const int global_num_rows = block_size*num_procs;
    const int first_row = block_size*rank;
    const int next_rank = (rank + 1) % num_procs;
    const int next_row = block_size*next_rank;

    ParCSRMatrix* A_csr = new ParCSRMatrix(global_num_rows, global_num_rows);
    ASSERT_EQ(A_csr->local_num_rows, block_size);
    // each rank owns the following
    A_csr->add_value(0, first_row, 4.0);
    A_csr->add_value(0, first_row + 1, 1.0);
    A_csr->add_value(0, next_row, -1.0);
    A_csr->add_value(1, first_row, 2.0);
    A_csr->add_value(1, first_row + 1, 3.0);
    A_csr->add_value(1, next_row + 1, -2.0);
    A_csr->on_proc->idx1[1] = 2;
    A_csr->on_proc->idx1[2] = 4;
    A_csr->off_proc->idx1[1] = 1;
    A_csr->off_proc->idx1[2] = 2;
    A_csr->finalize();

    // identity T
    ParCSRMatrix* T_csr = new ParCSRMatrix(global_num_rows, global_num_rows);
    T_csr->add_value(0, first_row, 1.0);
    T_csr->add_value(1, first_row + 1, 1.0);
    T_csr->on_proc->idx1[1] = 1;
    T_csr->on_proc->idx1[2] = 2;
    T_csr->finalize();

    ParBSRMatrix* A = A_csr->to_ParBSR(block_size, block_size);
    ParBSRMatrix* T = T_csr->to_ParBSR(block_size, block_size);
    delete A_csr;
    delete T_csr;

    ParBSRMatrix* P = jacobi_prolongation(A, T, false, 0.5, 1, prolongation_weighting::block);
    P->sort();

    BSRMatrix* P_on = static_cast<BSRMatrix*>(P->on_proc);
    BSRMatrix* P_off = static_cast<BSRMatrix*>(P->off_proc);
    ASSERT_EQ(P->local_num_rows, 1);
    ASSERT_EQ(P_on->nnz, 1);
    ASSERT_EQ(P_off->nnz, 1);

    const double* local_block = P_on->block_vals[0];
    EXPECT_NEAR(local_block[0], 0.5, 1e-12);
    EXPECT_NEAR(local_block[1], 0.0, 1e-12);
    EXPECT_NEAR(local_block[2], 0.0, 1e-12);
    EXPECT_NEAR(local_block[3], 0.5, 1e-12);

    ASSERT_EQ(P->off_proc_column_map[P_off->idx2[0]], next_rank);
    const double* remote_block = P_off->block_vals[0];
    EXPECT_NEAR(remote_block[0], 0.15, 1e-12);
    EXPECT_NEAR(remote_block[1], -0.1, 1e-12);
    EXPECT_NEAR(remote_block[2], -0.1, 1e-12);
    EXPECT_NEAR(remote_block[3], 0.4, 1e-12);

    delete P;
    delete T;
    delete A;
}
