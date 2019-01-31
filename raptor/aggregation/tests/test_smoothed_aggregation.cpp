// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "raptor.hpp"
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
    CSRMatrix* T;
    CSRMatrix* P;
    CSRMatrix* Ac;
    aligned_vector<int> states;
    aligned_vector<int> aggs;
    aligned_vector<double> weights;

    CSRMatrix* S_py;
    CSRMatrix* T_py;
    CSRMatrix* P_py;
    CSRMatrix* Ac_py;
    aligned_vector<int> py_states;
    aligned_vector<int> py_aggs;

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

    A = readMatrix(A0_fn);
    weights.resize(A->n_rows);
    py_states.resize(A->n_rows);
    py_aggs.resize(A->n_rows);

    f = fopen(weights_fn, "r");
    for (int i = 0; i < A->n_rows; i++)
    {
        fscanf(f, "%lf\n", &weights[i]);
    }
    fclose(f);

    // Test strength of connection
    S = A->strength(Symmetric, 0.25);
    S_py = readMatrix(S0_fn);
    compare_pattern(S, S_py);
    delete S_py;

    // Test MIS2
    f = fopen(mis0_fn, "r");
    for (int i = 0; i < A->n_rows; i++)
    {
        fscanf(f, "%d\n", &py_states[i]);
    }
    fclose(f);
    mis2(S, states, weights.data());
    for (int i = 0; i < A->n_rows; i++)
    {
        ASSERT_EQ(states[i], py_states[i]);
    }

    // Test aggregates
    f = fopen(agg0_fn, "r");
    for (int i = 0; i < A->n_rows; i++)
    {
        fscanf(f, "%d\n", &py_aggs[i]);
    }
    fclose(f);
    int n_aggs = aggregate(A, S, states, aggs, weights.data());
    for (int i = 0; i < A->n_rows; i++)
    {
        ASSERT_EQ(aggs[i], py_aggs[i]);
    }

    // Test tentative interpolation
    int num_candidates = 1;
    aligned_vector<double> B;
    aligned_vector<double> R;
    B.resize(A->n_rows, 1.0);
    T_py = readMatrix(T0_fn);
    T = fit_candidates(n_aggs, aggs, B, R, num_candidates, 1e-10);
    compare(T, T_py);
    delete T_py;

    // Test jacobi prolongation smoothing
    P_py = readMatrix(P0_fn);
    P = jacobi_prolongation(A, T);
    compare(P, P_py);
    delete P_py;

    // Compare RAP
    Ac_py = readMatrix(A1_fn);
    CSRMatrix* AP = A->mult(P);
    CSCMatrix* P_csc = P->to_CSC();
    Ac = AP->mult_T(P_csc);    
    compare(Ac, Ac_py);
    delete Ac_py;
    delete P;
    delete T;
    delete S;
    delete A;

    A = Ac;
    Ac = NULL;


    // Level 1
    
    // Test strength of connection
    S = A->strength(Symmetric, 0.25);
    S_py = readMatrix(S1_fn);
    compare_pattern(S, S_py);
    delete S_py;

    // Test MIS2
    f = fopen(mis1_fn, "r");
    for (int i = 0; i < A->n_rows; i++)
    {
        fscanf(f, "%d\n", &py_states[i]);
    }
    fclose(f);
    mis2(S, states, weights.data());
    for (int i = 0; i < A->n_rows; i++)
    {
        ASSERT_EQ(states[i], py_states[i]);
    }

    // Test aggregates
    f = fopen(agg1_fn, "r");
    for (int i = 0; i < A->n_rows; i++)
    {
        fscanf(f, "%d\n", &py_aggs[i]);
    }
    fclose(f);
    n_aggs = aggregate(A, S, states, aggs, weights.data());
    for (int i = 0; i < A->n_rows; i++)
    {
        ASSERT_EQ(aggs[i], py_aggs[i]);
    }

    // Test tentative interpolation
    B.resize(A->n_rows, 1.0);
    T_py = readMatrix(T1_fn);
    T = fit_candidates(n_aggs, aggs, B, R, num_candidates, 1e-10);
    compare(T, T_py);
    delete T_py;

    // Test jacobi prolongation smoothing
    P_py = readMatrix(P1_fn);
    P = jacobi_prolongation(A, T);
    P->sort();
    P->move_diag();
    P_py->sort();
    P_py->move_diag();
    compare(P, P_py);
    delete P_py;


} // end of TEST(TestSplitting, TestsInRuge_Stuben) //


