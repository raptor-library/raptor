// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
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

    // TEST LEVEL 0
    A = readMatrix("../../../../test_data/rss_A0.mtx", 1);
    P = readMatrix("../../../../test_data/rss_P0.mtx", 0);
    AP = A->mult(P);
    P_csc = new CSCMatrix(P);
    Ac = AP->mult_T(P_csc);
    Ac_rap = readMatrix("../../../../test_data/rss_A1.mtx", 0);
    compare(Ac, Ac_rap);
    delete Ac_rap;
    delete Ac;
    delete P_csc;
    delete AP;
    delete P;
    delete A;

    // TEST LEVEL 1
    A = readMatrix("../../../../test_data/rss_A1.mtx", 0);
    P = readMatrix("../../../../test_data/rss_P1.mtx", 0);
    AP = A->mult(P);
    P_csc = new CSCMatrix(P);
    Ac = AP->mult_T(P_csc);
    Ac_rap = readMatrix("../../../../test_data/rss_A2.mtx", 0);
    compare(Ac, Ac_rap);
    delete Ac_rap;
    delete Ac;
    delete P_csc;
    delete AP;
    delete P;
    delete A;
} // end of TEST(TestRAP, TestsInRuge_Stuben) //
