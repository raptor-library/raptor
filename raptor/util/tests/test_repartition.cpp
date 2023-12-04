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


TEST(Repartition, TestsInUtil)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int* proc_part;
    std::vector<int> new_local_rows;

    const char* filename = "../../../../test_data/random.pm";
    ParCSRMatrix* A_orig = readParMatrix(filename);
    ParVector x_orig(A_orig->global_num_rows, A_orig->local_num_rows);
    ParVector b_orig(A_orig->global_num_rows, A_orig->local_num_rows);
    A_orig->tap_comm = new TAPComm(A_orig->partition, A_orig->off_proc_column_map);
    for (int i = 0; i < A_orig->local_num_rows; i++)
    {
        x_orig[i] = A_orig->on_proc_column_map[i];
    }


    // RoundRobin Partitioning
    proc_part = new int[A_orig->local_num_rows];
    for (int i = 0; i < A_orig->local_num_rows; i++)
    {
        proc_part[i] = A_orig->local_row_map[i] % num_procs;
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
    x_orig.set_const_value(1.0);
    A_orig->mult(x_orig, b_orig);
    std::vector<int> row_sizes(num_procs);
    std::vector<int> row_displs(num_procs+1);
    MPI_Allgather(&(A_orig->local_num_rows), 1, MPI_INT, row_sizes.data(), 1, 
            MPI_INT, MPI_COMM_WORLD);
    row_displs[0] = 0;
    for (int i = 0; i < num_procs; i++)
        row_displs[i+1] = row_displs[i] + row_sizes[i];
    std::vector<double> orig_sol(A_orig->global_num_rows);
    MPI_Gatherv(b_orig.local.data(), A_orig->local_num_rows, MPI_DOUBLE, 
           orig_sol.data(), row_sizes.data(), row_displs.data(), 
           MPI_DOUBLE, 0, MPI_COMM_WORLD);
    x_rr.set_const_value(1.0);
    A_rr->mult(x_rr, b_rr);
    MPI_Allgather(&(A_rr->local_num_rows), 1, MPI_INT, row_sizes.data(), 1, 
            MPI_INT, MPI_COMM_WORLD);
    row_displs[0] = 0;
    for (int i = 0; i < num_procs; i++)
        row_displs[i+1] = row_displs[i] + row_sizes[i];
    std::vector<double> rr_sol(A_rr->global_num_rows);
    MPI_Gatherv(b_rr.local.data(), A_rr->local_num_rows, MPI_DOUBLE, 
           rr_sol.data(), row_sizes.data(), row_displs.data(), 
           MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::vector<int> orig_to_rr(A_orig->global_num_rows);
    MPI_Gatherv(new_local_rows.data(), A_rr->local_num_rows, MPI_INT, 
           orig_to_rr.data(), row_sizes.data(), row_displs.data(), 
           MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) 
    {
        for (int i = 0; i < A_orig->global_num_rows; i++)
        {
            ASSERT_NEAR(rr_sol[i], orig_sol[orig_to_rr[i]], 1e-06);
        }
    }

    delete A_rr;

    delete A_orig;

}

