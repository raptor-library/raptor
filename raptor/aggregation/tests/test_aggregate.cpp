// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "raptor.hpp"

#include <iostream>
#include <fstream>

using namespace raptor;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} // end of main() //


TEST(TestAggregate, TestsInAggregation)
{ 
    FILE* f;
    CSRMatrix* A;
    CSRMatrix* S;

    const char* A0_fn = "../../../../test_data/sas_A0.pm";
    const char* S0_fn = "../../../../test_data/sas_S0.pm";
    const char* mis0_fn = "../../../../test_data/sas_mis0.txt";
    const char* agg0_fn = "../../../../test_data/sas_agg0.txt";
    const char* weights_fn = "../../../../test_data/weights.txt";

    A = readMatrix(A0_fn);
    S = readMatrix(S0_fn);
    int n_items_read;

    f = fopen(weights_fn, "r");
    aligned_vector<double> weights(S->n_rows);
    for (int i = 0; i < S->n_rows; i++)
    {
        n_items_read = fscanf(f, "%lf\n", &weights[i]);
    }
    fclose(f);

    aligned_vector<int> python_states(S->n_rows);
    f = fopen(mis0_fn, "r");
    for (int i = 0; i < S->n_rows; i++)
    {
        n_items_read = fscanf(f, "%d\n", &python_states[i]);
    }
    fclose(f);

    aligned_vector<int> python_aggs(S->n_rows);
    f = fopen(agg0_fn, "r");
    for (int i = 0; i < S->n_rows; i++)
    {
        n_items_read = fscanf(f, "%d\n", &python_aggs[i]);
    }
    fclose(f);


    aligned_vector<int> aggregates;
    int n_aggs =  aggregate(A, S, python_states, aggregates, weights.data());

    for (int i = 0; i < S->n_rows; i++)
    {
        ASSERT_EQ(aggregates[i], python_aggs[i]);
    }

    delete S;
    delete A;



    const char* A1_fn = "../../../../test_data/sas_A1.pm";
    const char* S1_fn = "../../../../test_data/sas_S1.pm";
    const char* mis1_fn = "../../../../test_data/sas_mis1.txt";
    const char* agg1_fn = "../../../../test_data/sas_agg1.txt";

    A = readMatrix(A1_fn);
    S = readMatrix(S1_fn);

    python_states.resize(S->n_rows);
    f = fopen(mis1_fn, "r");
    for (int i = 0; i < S->n_rows; i++)
    {
        n_items_read = fscanf(f, "%d\n", &python_states[i]);
    }
    fclose(f);

    python_aggs.resize(S->n_rows);
    f = fopen(agg1_fn, "r");
    for (int i = 0; i < S->n_rows; i++)
    {
        n_items_read = fscanf(f, "%d\n", &python_aggs[i]);
    }
    fclose(f);


    n_aggs =  aggregate(A, S, python_states, aggregates, weights.data());

    for (int i = 0; i < S->n_rows; i++)
    {
        ASSERT_EQ(aggregates[i], python_aggs[i]);
    }

    delete S;
    delete A;


} // end of TEST(TestSplitting, TestsInRuge_Stuben) //
