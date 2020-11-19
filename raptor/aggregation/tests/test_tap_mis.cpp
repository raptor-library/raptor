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

TEST(TestTAPMIS, TestsInAggregation)
{ 
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    setenv("PPN", "4", 1);

    FILE* f;
    std::vector<int> states;
    std::vector<int> off_proc_states;

    ParCSRMatrix* S;
    int n_items_read;

    const char* S0_fn = "../../../../test_data/sas_S0.pm";
    const char* mis0_fn = "../../../../test_data/sas_mis0.txt";
    const char* weights_fn = "../../../../test_data/weights.txt";

    S = readParMatrix(S0_fn);
    S->tap_comm = new TAPComm(S->partition, S->off_proc_column_map,
            S->on_proc_column_map);

    f = fopen(weights_fn, "r");
    std::vector<double> weights(S->local_num_rows);
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

    std::vector<int> python_states(S->local_num_rows);
    f = fopen(mis0_fn, "r");
    for (int i = 0; i < S->partition->first_local_row; i++)
    {
        n_items_read = fscanf(f, "%d\n", &python_states[0]);
        ASSERT_EQ(n_items_read, 1);
    }
    for (int i = 0; i < S->local_num_rows; i++)
    {
        n_items_read = fscanf(f, "%d\n", &python_states[i]);
        ASSERT_EQ(n_items_read, 1);
    }
    fclose(f);

    mis2(S, states, off_proc_states, true, weights.data());

    for (int i = 0; i < S->local_num_rows; i++)
    {
        ASSERT_EQ(states[i], python_states[i]);
    }

    setenv("PPN", "16", 1);    
    
    delete S;

} // end of TEST(TestParSplitting, TestsInRuge_Stuben) //



