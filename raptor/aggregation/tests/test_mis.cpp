// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "raptor/raptor.hpp"

#include <iostream>
#include <fstream>

using namespace raptor;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} // end of main() //


TEST(TestMIS, TestsInAggregation)
{ 
    FILE* f;
    CSRMatrix* S;

    const char* S0_fn = "../../../../test_data/sas_S0.pm";
    const char* mis0_fn = "../../../../test_data/sas_mis0.txt";
    const char* weights_fn = "../../../../test_data/weights.txt";

    S = readMatrix(S0_fn);
    int n_items_read;

    f = fopen(weights_fn, "r");
    std::vector<double> weights(S->n_rows);
    for (int i = 0; i < S->n_rows; i++)
    {
        n_items_read = fscanf(f, "%lf\n", &weights[i]);
        ASSERT_EQ(n_items_read, 1);
    }
    fclose(f);

    std::vector<int> python_states(S->n_rows);
    f = fopen(mis0_fn, "r");
    for (int i = 0; i < S->n_rows; i++)
    {
        n_items_read = fscanf(f, "%d\n", &python_states[i]);
        ASSERT_EQ(n_items_read, 1);
    }
    fclose(f);
    
    std::vector<int> states;
    mis2(S, states, weights.data());

    for (int i = 0; i < S->n_rows; i++)
    {
        ASSERT_EQ(states[i], python_states[i]);
    }

    delete S;

    const char* S1_fn = "../../../../test_data/sas_S1.pm";
    const char* mis1_fn = "../../../../test_data/sas_mis1.txt";

    S = readMatrix(S1_fn);
    python_states.resize(S->n_rows);
    f = fopen(mis1_fn, "r");
    for (int i = 0; i < S->n_rows; i++)
    {
        n_items_read = fscanf(f, "%d\n", &python_states[i]);
        ASSERT_EQ(n_items_read, 1);
    }
    fclose(f);
    
    states.clear();
    mis2(S, states, weights.data());

    for (int i = 0; i < S->n_rows; i++)
    {
        ASSERT_EQ(states[i], python_states[i]);
    }

    delete S;


} // end of TEST(TestSplitting, TestsInRuge_Stuben) //

