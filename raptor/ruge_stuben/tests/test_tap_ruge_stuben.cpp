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

TEST(TestTAPRugeStuben, TestsInRuge_Stuben)
{ 
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    setenv("PPN", "4", 1);

    ParCSRMatrix* A;
    ParCSRMatrix* S;
    ParCSRMatrix* P;
    ParCSCMatrix* P_csc;
    ParCSRMatrix* AP;
    ParCSRMatrix* Ac;
    ParCSRMatrix* Ac_rap;
    std::vector<int> proc_sizes(num_procs);
    std::vector<int> splitting;
    std::vector<int> off_proc_splitting;
    std::vector<double> rand_vals;
    int first_row;
    int n_items_read;

    // Read in weights
    int max_n = 5000;
    std::vector<double> weights(max_n);
    const char* weights_fn = "../../../../test_data/weights.txt";
    FILE* f = fopen(weights_fn, "r");
    for (int i = 0; i < max_n; i++)
    {
        n_items_read = fscanf(f, "%lf\n", &weights[i]);
        ASSERT_EQ(n_items_read, 1);
    }
    fclose(f);

    const char* A0_fn = "../../../../test_data/rss_A0.pm";
    const char* A1_fn = "../../../../test_data/rss_A1.pm";
    const char* A2_fn = "../../../../test_data/rss_A2.pm";

    // Test Level 0
    A = readParMatrix(A0_fn);
    A->init_tap_communicators();
    S = A->strength(Classical, 0.25, true);
    MPI_Allgather(&A->local_num_rows, 1, MPI_INT, proc_sizes.data(),
            1, MPI_INT, MPI_COMM_WORLD);
    first_row = 0;
    for (int i = 0; i < rank; i++)
    {
        first_row += proc_sizes[i];
    }
    rand_vals.resize(A->local_num_rows);
    for (int i = 0; i < A->local_num_rows; i++)
    {
        rand_vals[i] = weights[i + first_row];
    }
    // TODO -- work on adding TAP communication to CLJP effectively
    split_cljp(S, splitting, off_proc_splitting, true, rand_vals.data());
    P = direct_interpolation(A, S, splitting, off_proc_splitting);
    MPI_Allgather(&P->on_proc_num_cols, 1, MPI_INT, proc_sizes.data(),
            1, MPI_INT, MPI_COMM_WORLD);
    first_row = 0;
    for (int i = 0; i < rank; i++)
    {
        first_row += proc_sizes[i];
    }
    AP = A->tap_mult(P);
    P_csc = P->to_ParCSC();
    Ac_rap = AP->tap_mult_T(P_csc);
    Ac = readParMatrix(A1_fn, Ac_rap->local_num_rows,
            Ac_rap->on_proc_num_cols, first_row, first_row);
    compare(Ac, Ac_rap);

    delete Ac;
    delete P_csc;
    delete AP;
    delete P;
    delete S;
    delete A;

    // Test Level 1
    A = Ac_rap;
    A->init_tap_communicators();
    A->comm = new ParComm(A->partition, A->off_proc_column_map, A->on_proc_column_map);
    Ac_rap = NULL;
    S = A->strength(Classical, 0.25, true);
    rand_vals.resize(A->local_num_rows);
    for (int i = 0; i < A->local_num_rows; i++)
    {
        rand_vals[i] = weights[i + first_row];
    }
    split_cljp(S, splitting, off_proc_splitting, true, rand_vals.data());
    P = direct_interpolation(A, S, splitting, off_proc_splitting);
    MPI_Allgather(&P->on_proc_num_cols, 1, MPI_INT, proc_sizes.data(),
            1, MPI_INT, MPI_COMM_WORLD);
    first_row = 0;
    for (int i = 0; i < rank; i++)
    {
        first_row += proc_sizes[i];
    }
    AP = A->tap_mult(P);
    P_csc = P->to_ParCSC();
    Ac_rap = AP->tap_mult_T(P_csc);
    Ac = readParMatrix(A2_fn, Ac_rap->local_num_rows,
            Ac_rap->on_proc_num_cols, first_row, first_row);
    compare(Ac, Ac_rap);

    delete Ac_rap;
    delete Ac;
    delete P_csc;
    delete AP;
    delete P;
    delete S;
    delete A;

    setenv("PPN", "16", 1);

} // end of TEST(TestParRugeStuben, TestsInRuge_Stuben) //

