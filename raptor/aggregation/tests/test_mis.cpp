// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "gallery/matrix_IO.hpp"
#include "aggregation/mis.hpp"

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

    f = fopen(weights_fn, "r");
    aligned_vector<double> weights(S->n_rows);
    for (int i = 0; i < S->n_rows; i++)
    {
        fscanf(f, "%lf\n", &weights[i]);
    }
    fclose(f);

    aligned_vector<int> python_states(S->n_rows);
    f = fopen(mis0_fn, "r");
    for (int i = 0; i < S->n_rows; i++)
    {
        fscanf(f, "%d\n", &python_states[i]);
    }
    fclose(f);
    
    aligned_vector<int> states;
    mis2(S, states, weights.data());

    for (int i = 0; i < S->n_rows; i++)
    {
        ASSERT_EQ(states[i], python_states[i]);
    }

    delete S;

} // end of TEST(TestSplitting, TestsInRuge_Stuben) //

