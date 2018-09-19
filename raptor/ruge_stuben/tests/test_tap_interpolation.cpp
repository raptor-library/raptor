// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "ruge_stuben/par_cf_splitting.hpp"
#include "ruge_stuben/par_interpolation.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/par_stencil.hpp"
#include "tests/par_compare.hpp"
#include <iostream>
#include <fstream>

using namespace raptor;

ParCSRMatrix* form_Prap(ParCSRMatrix* A, ParCSRMatrix* S, const char* filename, int* first_row_ptr, int* first_col_ptr, int interp_option = 0)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int first_row, first_col;
    FILE* f;
    ParCSRMatrix* P_rap = NULL;
    aligned_vector<int> proc_sizes(num_procs);
    aligned_vector<int> splitting;
    if (A->local_num_rows)
    {
        splitting.resize(A->local_num_rows);
    }
    MPI_Allgather(&A->local_num_rows, 1, MPI_INT, proc_sizes.data(), 1, MPI_INT,
            MPI_COMM_WORLD);
    first_row = 0;
    for (int i = 0; i < rank; i++)
    {
        first_row += proc_sizes[i];
    }
    f = fopen(filename, "r");
    int cf;
    for (int i = 0; i < first_row; i++)
    {
        fscanf(f, "%d\n", &cf);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        fscanf(f, "%d\n", &splitting[i]);
    }
    fclose(f);

    // Get off proc states
    S->comm->communicate(splitting.data());
    if (interp_option == 0)
    {
        P_rap = direct_interpolation(A, S, splitting, S->comm->recv_data->int_buffer);
    }
    else if (interp_option == 1)
    {
        P_rap = mod_classical_interpolation(A, S, splitting, S->comm->recv_data->int_buffer, true);
        return P_rap;
    }
    else if (interp_option == 2)
    {
        P_rap = extended_interpolation(A, S, splitting, S->comm->recv_data->int_buffer, true);
    }
    MPI_Allgather(&P_rap->on_proc_num_cols, 1, MPI_INT, proc_sizes.data(), 1, 
                MPI_INT, MPI_COMM_WORLD);
    first_col = 0;
    for (int i = 0; i < rank; i++)
    {
        first_col += proc_sizes[i];
    }

    *first_row_ptr = first_row;
    *first_col_ptr = first_col;

    return P_rap;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp = RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //


TEST(TestTAPInterpolation, TestsInRuge_Stuben)
{ 
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    setenv("PPN", "4", 1);

    int first_row, first_col, col;
    int start, end;
    FILE* f;
    ParCSRMatrix* A;
    ParCSRMatrix* S;
    ParCSRMatrix* P;
    ParCSRMatrix* P_rap;

    const char* A0_fn = "../../../../test_data/rss_A0.pm";
    const char* A1_fn = "../../../../test_data/rss_A1.pm";
    const char* S0_fn = "../../../../test_data/rss_S0.pm";
    const char* S1_fn = "../../../../test_data/rss_S1.pm";
    const char* P0_fn = "../../../../test_data/rss_P0.pm";
    const char* P1_fn = "../../../../test_data/rss_P1.pm";
    const char* cf0_fn = "../../../../test_data/rss_cf0.txt";
    const char* cf1_fn = "../../../../test_data/rss_cf1.txt";
    const char* P0_mc_fn = "../../../../test_data/rss_P0_mc.pm";
    const char* P1_mc_fn = "../../../../test_data/rss_P1_mc.pm";
    const char* P0_extend = "../../../../test_data/rss_P0_extend.pm";
    const char* P1_extend = "../../../../test_data/rss_P1_extend.pm";

    // TEST LEVEL 0
    A = readParMatrix(A0_fn);
    A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map);
    S = readParMatrix(S0_fn);
    aligned_vector<int> A_to_S(A->off_proc_num_cols, -1);
    int ctr = 0;
    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        int global_col = S->off_proc_column_map[i];
        while (A->off_proc_column_map[ctr] != global_col)
        ctr++;
        A_to_S[ctr] = i;
    }
    if (!S->tap_comm) 
        S->tap_comm = new TAPComm((TAPComm*) A->tap_comm, A_to_S);
    if (!S->comm)
        S->comm = new ParComm((ParComm*) A->comm, A_to_S);

    P_rap = form_Prap(A, S, cf0_fn, 
            &first_row, &first_col, 0);
    P = readParMatrix(P0_fn, P_rap->local_num_rows, P_rap->on_proc_num_cols, 
            first_row, first_col);
    compare(P, P_rap);
    delete P_rap;
    delete P;

    P_rap = form_Prap(A, S, cf0_fn, 
            &first_row, &first_col, 1);
    P = readParMatrix(P0_mc_fn, P_rap->local_num_rows, P_rap->on_proc_num_cols, 
            first_row, first_col);
    compare(P, P_rap);
    delete P;
    delete P_rap;

    P_rap = form_Prap(A, S, cf0_fn, &first_row, &first_col, 2);
    P = readParMatrix(P0_extend, P_rap->local_num_rows, P_rap->on_proc_num_cols, 
            first_row, first_col);
    compare(P, P_rap);
    delete P;
    delete P_rap;

    delete S;
    delete A;

    // TEST LEVEL 1
    A = readParMatrix(A1_fn);
    A->tap_comm = new TAPComm(A->partition, 
            A->off_proc_column_map, A->on_proc_column_map);
    S = readParMatrix(S1_fn);
    A_to_S.resize(A->off_proc_num_cols);
    std::fill(A_to_S.begin(), A_to_S.end(), -1);
    ctr = 0;
    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        int global_col = S->off_proc_column_map[i];
        while (A->off_proc_column_map[ctr] != global_col)
        ctr++;
        A_to_S[ctr] = i;
    }
    if (!S->tap_comm) 
        S->tap_comm = new TAPComm((TAPComm*) A->tap_comm, A_to_S);
    if (!S->comm)
        S->comm = new ParComm((ParComm*) A->comm, A_to_S);



    P_rap = form_Prap(A, S, cf1_fn, 
            &first_row, &first_col, 0);
    P = readParMatrix(P1_fn, P_rap->local_num_rows, P_rap->on_proc_num_cols, 
            first_row, first_col);
    compare(P, P_rap);
    delete P_rap;
    delete P;

    P_rap = form_Prap(A, S, cf1_fn, &first_row, &first_col, 1);
    P = readParMatrix(P1_mc_fn, P_rap->local_num_rows, P_rap->on_proc_num_cols, 
            first_row, first_col);
    P->sort();
    P_rap->sort();
    compare(P, P_rap);
    delete P;
    delete P_rap;

    P_rap = form_Prap(A, S, cf1_fn, &first_row, &first_col, 2);
    P = readParMatrix(P1_extend, P_rap->local_num_rows, P_rap->on_proc_num_cols, 
            first_row, first_col);
    P->sort();
    P_rap->sort();
    compare(P, P_rap);
    delete P;
    delete P_rap;

    delete S;
    delete A;

    setenv("PPN", "16", 1);

} // end of TEST(TestParInterpolation, TestsInRuge_Stuben) //


