// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause


#include "gtest/gtest.h"
#include "raptor.hpp"
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

TEST(TestTAPSplitting, TestsInRuge_Stuben)
{ 
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    setenv("PPN", "4", 1);

    FILE* f;
    aligned_vector<int> states;
    aligned_vector<int> off_proc_states;
    int cf;

    ParCSRMatrix* S;

    const char* S0_fn = "../../../../test_data/rss_S0.pm";
    const char* S1_fn = "../../../../test_data/rss_S1.pm";
    const char* cf0_fn = "../../../../test_data/rss_cf0.txt";
    const char* cf1_fn = "../../../../test_data/rss_cf1.txt";
    const char* weights_fn = "../../../../test_data/weights.txt";
    int n_items_read;

    // TEST LEVEL 0
    S = readParMatrix(S0_fn);
    S->init_tap_communicators();

    f = fopen(weights_fn, "r");
    aligned_vector<double> weights(S->local_num_rows);
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
    split_cljp(S, states, off_proc_states, true, weights.data());
    
    f = fopen(cf0_fn, "r");
    for (int i = 0; i < S->partition->first_local_row; i++)
    {
        n_items_read = fscanf(f, "%d\n", &cf);
        ASSERT_EQ(n_items_read, 1);
    }
    for (int i = 0; i < S->local_num_rows; i++)
    {
        n_items_read = fscanf(f, "%d\n", &cf);
        ASSERT_EQ(n_items_read, 1);
        ASSERT_EQ(cf, states[i]);
    }
    fclose(f);

    delete S;

    // TEST LEVEL 1
    S = readParMatrix(S1_fn);
    S->init_tap_communicators();

    f = fopen(weights_fn, "r");
    weights.resize(S->local_num_rows);
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
    split_cljp(S, states, off_proc_states, true, weights.data());
    
    f = fopen(cf1_fn, "r");
    for (int i = 0; i < S->partition->first_local_row; i++)
    {
        n_items_read = fscanf(f, "%d\n", &cf);
        ASSERT_EQ(n_items_read, 1);
    }
    for (int i = 0; i < S->local_num_rows; i++)
    {
        n_items_read = fscanf(f, "%d\n", &cf);
        ASSERT_EQ(n_items_read, 1);
        ASSERT_EQ(cf, states[i]);
    }
    fclose(f);

    delete S;

    setenv("PPN", "16", 1);    
   

} // end of TEST(TestParSplitting, TestsInRuge_Stuben) //


