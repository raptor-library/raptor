// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "mpi.h"
#include "gallery/stencil.hpp"
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "aggregation/par_mis.hpp"
#include "aggregation/par_aggregate.hpp"
#include "aggregation/par_candidates.hpp"
#include "aggregation/par_prolongation.hpp"
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

TEST(TestParSmoothedAggregation, TestsInAggregation)
{ 
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    FILE* f;

    ParCSRMatrix* A;
    ParCSRMatrix* S;
    ParCSRMatrix* S_red;
    ParCSRMatrix* T;
    ParCSRMatrix* P;
    ParCSRMatrix* Ac;
    ParCSRMatrix* S_py;
    ParCSRMatrix* T_py;
    ParCSRMatrix* P_py;
    ParCSRMatrix* Ac_py;

    aligned_vector<int> states;
    aligned_vector<int> off_proc_states;
    aligned_vector<int> py_states;
    aligned_vector<int> aggs;
    aligned_vector<int> py_aggs;
    aligned_vector<double> weights;

    const char* A0_fn = "../../../../test_data/sas_A0.pm";
    const char* S0_fn = "../../../../test_data/sas_S0.pm";
    const char* mis0_fn = "../../../../test_data/sas_mis0.txt";
    const char* agg0_fn = "../../../../test_data/sas_agg0.txt";
    const char* T0_fn = "../../../../test_data/sas_T0.pm";
    const char* P0_fn = "../../../../test_data/sas_P0.pm";
    const char* A1_fn = "../../../../test_data/sas_A1.pm";
    const char* S1_fn = "../../../../test_data/sas_S1.pm";
    const char* mis1_fn = "../../../../test_data/sas_mis1.txt";
    const char* agg1_fn = "../../../../test_data/sas_agg1.txt";
    const char* T1_fn = "../../../../test_data/sas_T1.pm";
    const char* P1_fn = "../../../../test_data/sas_P1.pm";

    const char* weights_fn = "../../../../test_data/weights.txt";    

    A = readParMatrix(A0_fn);

    py_states.resize(A->local_num_rows);
    py_aggs.resize(A->local_num_rows);
    weights.resize(A->local_num_rows);

    f = fopen(weights_fn, "r");
    for (int i = 0; i < A->partition->first_local_row; i++)
    {
        fscanf(f, "%lf\n", &weights[0]);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        fscanf(f, "%lf\n", &weights[i]);
    }
    fclose(f);

    // Strength of connection
    S = A->strength(Symmetric, 0.25);
    S_py = readParMatrix(S0_fn);
    remove_empty_cols(S);
    compare_pattern(S, S_py);
    delete S_py;
    delete S;
    S = A->strength(Symmetric, 0.25);


    // Test MIS 2
    f = fopen(mis0_fn, "r");
    for (int i = 0; i < S->partition->first_local_row; i++)
    {
        fscanf(f, "%d\n", &py_states[0]);
    }
    for (int i = 0; i < S->local_num_rows; i++)
    {
        fscanf(f, "%d\n", &py_states[i]);
    }
    fclose(f);
    mis2(S, states, off_proc_states, false, weights.data());
    for (int i = 0; i < S->local_num_rows; i++)
    {
        ASSERT_EQ(states[i], py_states[i]);
    }

    // Test aggregates
    f = fopen(agg0_fn, "r");
    for (int i = 0; i < S->partition->first_local_row; i++)
    {
        fscanf(f, "%d\n", &py_aggs[0]);
    }
    for (int i = 0; i < S->local_num_rows; i++)
    {
        fscanf(f, "%d\n", &py_aggs[i]);
    }
    fclose(f);
    int n_aggs = aggregate(A, S, states, off_proc_states, 
            aggs, false, weights.data());

    // Aggregates returns global indices of original global rows
    // Gather list of all aggregates, in order, holding original global cols
    aligned_vector<int> agg_sizes(num_procs);
    aligned_vector<int> agg_displ(num_procs+1);
    aligned_vector<int> agg_list;
    aligned_vector<int> total_agg_list;
    int global_col, local_col;
    MPI_Allgather(&n_aggs, 1, MPI_INT, agg_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    agg_displ[0] = 0;
    int first_n = 0;
    for (int i = 0; i < num_procs; i++)
    {
        agg_displ[i+1] = agg_displ[i] + agg_sizes[i];
        if (i < rank) first_n += agg_sizes[i];
    }
    int total_size = agg_displ[num_procs];
    for (int i = 0; i < S->local_num_rows; i++)
    {
        if (states[i] > 0)
            agg_list.push_back(A->local_row_map[i]);
    }
    total_agg_list.resize(total_size);
    MPI_Allgatherv(agg_list.data(), n_aggs, MPI_INT, total_agg_list.data(), 
            agg_sizes.data(), agg_displ.data(), MPI_INT, MPI_COMM_WORLD);
    // Map aggregates[i] (global_col) to original global col that
    // now holds py_aggregates[i]
    for (int i = 0; i < A->local_num_rows; i++)
    {
        global_col = aggs[i];
        ASSERT_EQ(total_agg_list[py_aggs[i]], global_col);
    }

    // Test fitting candidates
    aligned_vector<int> proc_aggs(num_procs);
    int first_col = 0;
    MPI_Allgather(&n_aggs, 1, MPI_INT, proc_aggs.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int i = 0; i < rank; i++)
    {
        first_col += proc_aggs[i];
    }
    T_py = readParMatrix(T0_fn, A->local_num_rows, n_aggs, 
            A->partition->first_local_row, first_col);

    aligned_vector<double> B;
    aligned_vector<double> R;
    if (A->local_num_rows)
        B.resize(A->local_num_rows, 1.0);
    int num_candidates = 1;
    T = fit_candidates(A, n_aggs, aggs, B, R, num_candidates, false, 1e-10);
    compare(T, T_py); 
    delete T_py;


    // Test prolongation
    P_py = readParMatrix(P0_fn, A->local_num_rows, n_aggs,
            A->partition->first_local_row, first_col);
    P = jacobi_prolongation(A, T);
    compare(P, P_py);
    delete P_py;


    Ac_py = readParMatrix(A1_fn, n_aggs, n_aggs, first_col, first_col);
    ParCSRMatrix* AP = A->mult(P);
    ParCSCMatrix* P_csc = P->to_ParCSC();
    Ac = AP->mult_T(P_csc);    
    compare(Ac, Ac_py);
    delete Ac_py;
    delete AP;
    delete P_csc;

    delete P;
    delete T;
    delete S;
    delete A;

    A = Ac;
    A->comm = new ParComm(A->partition, A->off_proc_column_map,
            A->on_proc_column_map);
    Ac = NULL;

    // Test Level 1
    f = fopen(weights_fn, "r");
    for (int i = 0; i < first_col; i++)
    {
        fscanf(f, "%lf\n", &weights[0]);
    }
    for (int i = 0; i < n_aggs; i++)
    {
        fscanf(f, "%lf\n", &weights[i]);
    }
    fclose(f);

    // Strength of connection
    S_py = readParMatrix(S1_fn, n_aggs, n_aggs, first_col, first_col);
    S = A->strength(Symmetric, 0.25);
    remove_empty_cols(S);
    compare_pattern(S, S_py);
    delete S;
    delete S_py;
    S = A->strength(Symmetric, 0.25);

    // MIS2
    f = fopen(mis1_fn, "r");
    for (int i = 0; i < first_col; i++)
    {
        fscanf(f, "%d\n", &py_states[0]);
    }
    for (int i = 0; i < n_aggs; i++)
    {
        fscanf(f, "%d\n", &py_states[i]);
    }
    fclose(f);
    mis2(S, states, off_proc_states, false, weights.data());
    for (int i = 0; i < n_aggs; i++)
    {
        ASSERT_EQ(states[i], py_states[i]);
    }

    // Test aggregates
    f = fopen(agg1_fn, "r");
    for (int i = 0; i < first_col; i++)
    {
        fscanf(f, "%d\n", &py_aggs[0]);
    }
    for (int i = 0; i < n_aggs; i++)
    {
        fscanf(f, "%d\n", &py_aggs[i]);
    }
    fclose(f);
    aggs.clear();
    n_aggs = aggregate(A, S, states, off_proc_states, 
            aggs, false, weights.data());

    // Aggregates returns global indices of original global rows
    // Gather list of all aggregates, in order, holding original global cols
    MPI_Allgather(&n_aggs, 1, MPI_INT, agg_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    agg_displ[0] = 0;
    first_n = 0;
    for (int i = 0; i < num_procs; i++)
    {
        agg_displ[i+1] = agg_displ[i] + agg_sizes[i];
        if (i < rank) first_n += agg_sizes[i];
    }
    total_size = agg_displ[num_procs];
    agg_list.clear();
    for (int i = 0; i < S->local_num_rows; i++)
    {
        if (states[i] > 0)
            agg_list.push_back(A->local_row_map[i]);
    }
    total_agg_list.resize(total_size);
    MPI_Allgatherv(agg_list.data(), n_aggs, MPI_INT, total_agg_list.data(), 
            agg_sizes.data(), agg_displ.data(), MPI_INT, MPI_COMM_WORLD);
    // Map aggregates[i] (global_col) to original global col that
    // now holds py_aggregates[i]
    for (int i = 0; i < A->local_num_rows; i++)
    {
        global_col = aggs[i];
        ASSERT_EQ(total_agg_list[py_aggs[i]], global_col);
    }

    // Test fitting candidates
    int old_first_col = first_col;
    first_col = 0;
    MPI_Allgather(&n_aggs, 1, MPI_INT, proc_aggs.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int i = 0; i < rank; i++)
    {
        first_col += proc_aggs[i];
    }
    T_py = readParMatrix(T1_fn, A->local_num_rows, n_aggs, 
            old_first_col, first_col);
    if (A->local_num_rows)
        B.resize(A->local_num_rows, 1.0);
    T = fit_candidates(A, n_aggs, aggs, B, R, num_candidates, false, 1e-10);
    compare(T, T_py); 
    delete T_py;

    // Test prolongation
    P_py = readParMatrix(P1_fn, A->local_num_rows, n_aggs,
            old_first_col, first_col);
    P = jacobi_prolongation(A, T);
    compare(P, P_py);
    delete P_py;

    delete P;
    delete T;
    delete S;
    delete A;

} // end of TEST(TestParSplitting, TestsInRuge_Stuben) //



