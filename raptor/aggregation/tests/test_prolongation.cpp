// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "raptor/raptor.hpp"
#include "raptor/tests/compare.hpp"

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
    CSRMatrix* A;
    CSRMatrix* T;

    const char* A0_fn = "../../../../test_data/sas_A0.pm";
    const char* T0_fn = "../../../../test_data/sas_T0.pm";
    const char* P0_fn = "../../../../test_data/sas_P0.pm";

    const char* A1_fn = "../../../../test_data/sas_A1.pm";
    const char* T1_fn = "../../../../test_data/sas_T1.pm";
    const char* P1_fn = "../../../../test_data/sas_P1.pm";

    A = readMatrix(A0_fn);
    T = readMatrix(T0_fn);

    CSRMatrix* P_py = readMatrix(P0_fn);
    CSRMatrix* P = jacobi_prolongation(A, T);

    compare(P, P_py);

    delete P;
    delete P_py;
    delete T;
    delete A;

    A = readMatrix(A1_fn);
    T = readMatrix(T1_fn);

    P_py = readMatrix(P1_fn);
    P = jacobi_prolongation(A, T);
    P->sort();
    P->move_diag();
    P_py->sort();
    P_py->move_diag();

    compare(P, P_py);

    delete P;
    delete P_py;
    delete T;
    delete A;



} // end of TEST(TestSplitting, TestsInRuge_Stuben) //

