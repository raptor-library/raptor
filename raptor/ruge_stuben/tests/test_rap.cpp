// Copyright (c) 2015-2017, RAPtor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "gallery/matrix_IO.hpp"
#include "tests/compare.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} // end of main() //

TEST(TestRAP, TestsInRuge_Stuben)
{ 
    CSRMatrix* A;
    CSRMatrix* P;
    CSRMatrix* Ac;
    CSRMatrix* AP;
    CSCMatrix* P_csc;
    CSRMatrix* Ac_rap;

    const char* A0_fn = "../../../../test_data/rss_A0.mtx";
    const char* A1_fn = "../../../../test_data/rss_A1.mtx";
    const char* A2_fn = "../../../../test_data/rss_A2.mtx";
    const char* P0_fn = "../../../../test_data/rss_P0.mtx";
    const char* P1_fn = "../../../../test_data/rss_P1.mtx";


    // TEST LEVEL 0
    A = readMatrix(A0_fn, 1);
    P = readMatrix(P0_fn, 0);
    AP = A->mult(P);
    P_csc = new CSCMatrix(P);
    Ac = AP->mult_T(P_csc);
    Ac_rap = readMatrix(A1_fn, 0);
    compare(Ac, Ac_rap);
    delete Ac_rap;
    delete Ac;
    delete P_csc;
    delete AP;
    delete P;
    delete A;

    // TEST LEVEL 1
    A = readMatrix(A1_fn, 0);
    P = readMatrix(P1_fn, 0);
    AP = A->mult(P);
    P_csc = new CSCMatrix(P);
    Ac = AP->mult_T(P_csc);
    Ac_rap = readMatrix(A2_fn, 0);
    compare(Ac, Ac_rap);
    delete Ac_rap;
    delete Ac;
    delete P_csc;
    delete AP;
    delete P;
    delete A;
} // end of TEST(TestRAP, TestsInRuge_Stuben) //
