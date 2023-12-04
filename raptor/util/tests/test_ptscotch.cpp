// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "raptor/raptor.hpp" 

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp = RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //

TEST(ParMetisTest, TestsInUtil)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int* proc_part;
    double bnorm_orig, bnorm_rr, bnorm_part;

    const char* filename = "../../../../test_data/aniso.pm";
    int n_tests = 100;
    std::vector<int> new_local_rows;

    // Create RowWise Partition 
    ParCSRMatrix* A_orig = readParMatrix(filename);
    ParVector x_orig(A_orig->global_num_rows, A_orig->local_num_rows);
    ParVector b_orig(A_orig->global_num_rows, A_orig->local_num_rows);
    A_orig->tap_comm = new TAPComm(A_orig->partition, A_orig->off_proc_column_map);
    for (int i = 0; i < A_orig->local_num_rows; i++)
    {
        x_orig[i] = A_orig->on_proc_column_map[i];
    }
    A_orig->mult(x_orig, b_orig);

    // RoundRobin Partitioning
    proc_part = new int[A_orig->local_num_rows];
    for (int i = 0; i < A_orig->local_num_rows; i++)
    {
        proc_part[i] = i % num_procs;
    }
    ParCSRMatrix* A_rr = repartition_matrix(A_orig, proc_part, new_local_rows);
    ParVector x_rr(A_rr->global_num_rows, A_rr->local_num_rows);
    ParVector b_rr(A_rr->global_num_rows, A_rr->local_num_rows);
    A_rr->tap_comm = new TAPComm(A_rr->partition, A_rr->off_proc_column_map);
    for (int i = 0; i < A_rr->local_num_rows; i++)
    {
        x_rr[i] = new_local_rows[i];
    }
    delete[] proc_part;
    A_rr->mult(x_rr, b_rr);

    // Time Graph Partitioning
    proc_part = ptscotch_partition(A_orig);
    ParCSRMatrix* A = repartition_matrix(A_orig, proc_part, new_local_rows);
    delete[] proc_part;
    ParVector x(A->global_num_rows, A->local_num_rows);
    ParVector b(A->global_num_rows, A->local_num_rows);
    for (int i = 0; i < A->local_num_rows; i++)
    {
        x[i] = new_local_rows[i];
    }
    A->mult(x, b);

    bnorm_orig = b_orig.norm(2);
    bnorm_rr = b_rr.norm(2);
    bnorm_part = b.norm(2);

    ASSERT_NEAR(bnorm_orig, bnorm_rr, 1e-06);
    ASSERT_NEAR(bnorm_orig, bnorm_part, 1e-06);

    delete A_orig;
    delete A_rr;
    delete A;

} // end of TEST(ParMetisTest, TestsInUtil)

