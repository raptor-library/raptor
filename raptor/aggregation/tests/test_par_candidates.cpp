// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <iostream>
#include <fstream>
#include <cmath>

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

TEST(TestParCandidates, TestsInAggregation)
{ 
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    FILE* f;
    std::vector<int> states;
    std::vector<int> off_proc_states;

    ParCSRMatrix* A;
    ParCSRMatrix* S;
    int n_items_read;

    const char* A0_fn = "../../../../test_data/sas_A0.pm";
    const char* S0_fn = "../../../../test_data/sas_S0.pm";
    const char* T0_fn = "../../../../test_data/sas_T0.pm";
    const char* weights_fn = "../../../../test_data/weights.txt";

    A = readParMatrix(A0_fn);
    S = readParMatrix(S0_fn);

    std::vector<double> weights(S->local_num_rows);
    f = fopen(weights_fn, "r");
    for (int i = 0; i < S->partition->first_local_row; i++)
    {
        n_items_read = fscanf(f, "%lf\n", &weights[0]);
        ASSERT_EQ(n_items_read, 1);
    }
    for (int i = 0; i < S->local_num_rows; i++)
    {
        n_items_read = fscanf(f, "%lf\n", &weights[i]);
        ASSERT_EQ(n_items_read, 1);
    }
    fclose(f);

    mis2(S, states, off_proc_states, false, weights.data());
    std::vector<int> aggregates;
    int n_aggs = aggregate(A, S, states, off_proc_states, aggregates, 
            false, weights.data());

    std::vector<int> proc_aggs(num_procs);
    int first_col = 0;
    MPI_Allgather(&n_aggs, 1, MPI_INT, proc_aggs.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int i = 0; i < rank; i++)
    {
        first_col += proc_aggs[i];
    }

    ParCSRMatrix* T_py = readParMatrix(T0_fn, A->local_num_rows, n_aggs, 
            A->partition->first_local_row, first_col);

    std::vector<double> B;
    std::vector<double> R;
    if (A->local_num_rows)
        B.resize(A->local_num_rows, 1.0);
    int num_candidates = 1;
    ParCSRMatrix* T = fit_candidates(A, n_aggs, aggregates, B, R, num_candidates, false, 1e-10);

    compare(T, T_py); 

    delete T;
    delete T_py;
    delete A;
    delete S;

} // end of TEST(TestParSplitting, TestsInRuge_Stuben) //


TEST(TestParCandidates, FitDistributedCSRCandidates)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    if (num_procs < 2)
    {
        GTEST_SKIP() << "at least two MPI ranks required";
    }

    const int local_num_rows = 4;
    const int global_num_rows = local_num_rows*num_procs;
    const int num_candidates = 2;

    ParCSRMatrix* A = new ParCSRMatrix(global_num_rows, global_num_rows);
    A->finalize();
    ASSERT_EQ(A->local_num_rows, local_num_rows);

    // rank 0 owns the aggregate; others access it off-process
    const int n_aggs = (rank == 0) ? 1 : 0;
    std::vector<int> aggregates = {0, 0, 0, 0};

    // Candidate-major storage:
    // b_0 = [1, 0, 1, 0]^T and b_1 = [1, 1, 1, 1]^T locally
    std::vector<double> B = {
        1.0, 0.0, 1.0, 0.0,
        1.0, 1.0, 1.0, 1.0
    };
    std::vector<double> R;

    ParCSRMatrix* T = fit_candidates(A, n_aggs, aggregates, B, R,
            num_candidates, false, 1e-10);

    const double global_norm = std::sqrt(2.0*num_procs);
    const double inverse_global_norm = 1.0 / global_norm;
    const double tolerance = 1e-12;

    if (rank == 0)
    {
        ASSERT_EQ(R.size(), 4);
        EXPECT_NEAR(R[0], global_norm, tolerance);
        EXPECT_NEAR(R[1], global_norm, tolerance);
        EXPECT_NEAR(R[2], 0.0, tolerance);
        EXPECT_NEAR(R[3], global_norm, tolerance);
    }
    else
    {
        EXPECT_TRUE(R.empty());
    }

    CSRMatrix* T_on = static_cast<CSRMatrix*>(T->on_proc);
    CSRMatrix* T_off = static_cast<CSRMatrix*>(T->off_proc);
    CSRMatrix* T_local = (rank == 0) ? T_on : T_off;
    EXPECT_EQ(T->on_proc->nnz, (rank == 0) ? 2*local_num_rows : 0);
    EXPECT_EQ(T->off_proc->nnz, (rank == 0) ? 0 : 2*local_num_rows);
    ASSERT_EQ(T_local->nnz, 2*local_num_rows);

    for (int row = 0; row < local_num_rows; row++)
    {
        const int row_start = T_local->idx1[row];
        ASSERT_EQ(T_local->idx1[row + 1] - row_start, num_candidates);
        EXPECT_EQ(T_local->idx2[row_start], 0);
        EXPECT_EQ(T_local->idx2[row_start + 1], 1);

        const bool first_candidate_entry = (row % 2 == 0);
        EXPECT_NEAR(T_local->vals[row_start],
                first_candidate_entry ? inverse_global_norm : 0.0,
                tolerance);
        EXPECT_NEAR(T_local->vals[row_start + 1],
                first_candidate_entry ? 0.0 : inverse_global_norm,
                tolerance);
    }

    delete T;
    delete A;
} // end of TEST(TestParCandidates, FitsDistributedCSRCandidates) //


TEST(TestParCandidates, FitDistributedBSRCandidates)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    if (num_procs < 2)
    {
        GTEST_SKIP() << "at least two MPI ranks required";
    }

    const int block_size = 2;
    const int local_num_block_rows = 2;
    const int global_num_block_rows = local_num_block_rows*num_procs;
    const int num_candidates = 2;

    ParBSRMatrix* A = new ParBSRMatrix(global_num_block_rows,
            global_num_block_rows, block_size, block_size);
    A->finalize();
    ASSERT_EQ(A->local_num_rows, local_num_block_rows);

    // rank 0 owns the aggregate; others access it off-process
    const int n_aggs = (rank == 0) ? 1 : 0;
    std::vector<int> aggregates = {0, 0};

    // Candidate-major storage:
    // b_0 = [1, 0, 1, 0]^T and b_1 = [1, 1, 1, 1]^T
    std::vector<double> B = {
        1.0, 0.0, 1.0, 0.0,
        1.0, 1.0, 1.0, 1.0
    };
    std::vector<double> R;

    ParBSRMatrix* T = fit_candidates(A, n_aggs, aggregates, B, R,
            num_candidates, false, 1e-10);

    const double global_norm = std::sqrt(2.0*num_procs);
    const double inverse_global_norm = 1.0 / global_norm;
    const double tolerance = 1e-12;

    if (rank == 0)
    {
        ASSERT_EQ(R.size(), 4);
        EXPECT_NEAR(R[0], global_norm, tolerance);
        EXPECT_NEAR(R[1], global_norm, tolerance);
        EXPECT_NEAR(R[2], 0.0, tolerance);
        EXPECT_NEAR(R[3], global_norm, tolerance);
    }
    else
    {
        EXPECT_TRUE(R.empty());
    }

    BSRMatrix* T_on = static_cast<BSRMatrix*>(T->on_proc);
    BSRMatrix* T_off = static_cast<BSRMatrix*>(T->off_proc);
    BSRMatrix* T_local = (rank == 0) ? T_on : T_off;

    EXPECT_EQ(T_on->nnz, (rank == 0) ? local_num_block_rows : 0);
    EXPECT_EQ(T_off->nnz, (rank == 0) ? 0 : local_num_block_rows);
    ASSERT_EQ(T_local->b_rows, block_size);
    ASSERT_EQ(T_local->b_cols, num_candidates);
    ASSERT_EQ(T_local->nnz, local_num_block_rows);

    for (int i = 0; i < T_local->nnz; i++)
    {
        const double* block = T_local->block_vals[i];
        EXPECT_NEAR(block[0], inverse_global_norm, tolerance);
        EXPECT_NEAR(block[1], 0.0, tolerance);
        EXPECT_NEAR(block[2], 0.0, tolerance);
        EXPECT_NEAR(block[3], inverse_global_norm, tolerance);
    }

    delete T;
    delete A;
} // end of TEST(TestParCandidates, FitsDistributedBSRCandidates) //


