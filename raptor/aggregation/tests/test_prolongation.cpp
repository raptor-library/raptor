// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "gallery/matrix_IO.hpp"
#include "aggregation/prolongation.hpp"
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
    CSRMatrix* T;

    const char* A0_fn = "../../../../test_data/sas_A0.pm";
    const char* T0_fn = "../../../../test_data/sas_T0.pm";
    const char* P0_fn = "../../../../test_data/sas_P0.pm";

    A = readMatrix(A0_fn);
    T = readMatrix(T0_fn);

    CSRMatrix* P_py = readMatrix(P0_fn);
    CSRMatrix* P = jacobi_prolongation(A, T);

    compare(P, P_py);

    delete P;
    delete P_py;
    delete T;
    delete A;

} // end of TEST(TestSplitting, TestsInRuge_Stuben) //

