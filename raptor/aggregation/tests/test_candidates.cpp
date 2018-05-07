// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "gallery/matrix_IO.hpp"
#include "aggregation/aggregate.hpp"
#include "aggregation/candidates.hpp"
#include "tests/compare.hpp"

#include <iostream>
#include <fstream>

using namespace raptor;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} // end of main() //


TEST(TestCandidates, TestsInAggregation)
{ 
    FILE* f;
    CSRMatrix* A;
    CSRMatrix* S;

    const char* A0_fn = "../../../../test_data/sas_A0.pm";
    const char* S0_fn = "../../../../test_data/sas_S0.pm";
    const char* mis0_fn = "../../../../test_data/sas_mis0.txt";
    const char* T0_fn = "../../../../test_data/sas_T0.pm";

    A = readMatrix(A0_fn);
    S = readMatrix(S0_fn);

    aligned_vector<int> python_states(S->n_rows);
    f = fopen(mis0_fn, "r");
    for (int i = 0; i < S->n_rows; i++)
    {
        fscanf(f, "%d\n", &python_states[i]);
    }
    fclose(f);
    aligned_vector<int> aggregates;
    int n_aggs =  aggregate(A, S, python_states, aggregates);

    CSRMatrix* T_py = readMatrix(T0_fn);

    aligned_vector<double> B;
    aligned_vector<double> R;
    B.resize(A->n_rows, 1.0);
    int num_candidates = 1;

    CSRMatrix* T = fit_candidates(n_aggs, aggregates, B, R, 
            num_candidates, 1e-10);

    compare(T, T_py);

    delete T_py;
    delete T;
    delete S;
    delete A;

} // end of TEST(TestSplitting, TestsInRuge_Stuben) //

