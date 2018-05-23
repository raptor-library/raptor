// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "mpi.h"
#include "gallery/stencil.hpp"
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "aggregation/par_aggregate.hpp"
#include "aggregation/par_candidates.hpp"
#include "tests/par_compare.hpp"
#include <iostream>
#include <fstream>

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
    aligned_vector<int> states;
    aligned_vector<int> off_proc_states;

    ParCSRMatrix* A;
    ParCSRMatrix* S;

    const char* A0_fn = "../../../../test_data/sas_A0.pm";
    const char* S0_fn = "../../../../test_data/sas_S0.pm";
    const char* mis0_fn = "../../../../test_data/sas_mis0.txt";
    const char* T0_fn = "../../../../test_data/sas_T0.pm";
    const char* weights_fn = "../../../../test_data/weights.txt";

    A = readParMatrix(A0_fn);
    S = readParMatrix(S0_fn);

    aligned_vector<double> weights(S->local_num_rows);
    f = fopen(weights_fn, "r");
    for (int i = 0; i < S->partition->first_local_row; i++)
    {
        fscanf(f, "%lf\n", &weights[0]);
    }
    for (int i = 0; i < S->local_num_rows; i++)
    {
        fscanf(f, "%lf\n", &weights[i]);
    }
    fclose(f);

    mis2(S, states, off_proc_states, false, weights.data());
    aligned_vector<int> aggregates;
    int n_aggs = aggregate(A, S, states, off_proc_states, aggregates, 
            false, weights.data());

    aligned_vector<int> proc_aggs(num_procs);
    int first_col = 0;
    MPI_Allgather(&n_aggs, 1, MPI_INT, proc_aggs.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int i = 0; i < rank; i++)
    {
        first_col += proc_aggs[i];
    }

    ParCSRMatrix* T_py = readParMatrix(T0_fn, A->local_num_rows, n_aggs, 
            A->partition->first_local_row, first_col);

    aligned_vector<double> B;
    aligned_vector<double> R;
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






