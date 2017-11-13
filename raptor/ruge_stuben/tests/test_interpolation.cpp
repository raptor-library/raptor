// EXPECT_EQ and ASSERT_EQ are macros
// EXPECT_EQ test execution and continues even if there is a failure
// ASSERT_EQ test execution and aborts if there is a failure
// The ASSERT_* variants abort the program execution if an assertion fails 
// while EXPECT_* variants continue with the run.


#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "gallery/matrix_IO.hpp"
#include "ruge_stuben/cf_splitting.hpp"
#include "ruge_stuben/interpolation.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/stencil.hpp"
#include "tests/compare.hpp"

#include <iostream>
#include <fstream>

using namespace raptor;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} // end of main() //

TEST(TestInterpolation, TestsInRuge_Stuben)
{ 

    CSRMatrix* A;
    CSRMatrix* S;
    CSRMatrix* P;
    CSRMatrix* P_rap;
    std::vector<int> splitting;
    FILE* f;

    // TEST LEVEL 0
    A = readMatrix("../../../../test_data/rss_A0.mtx", 1);
    S = readMatrix("../../../../test_data/rss_S0.mtx", 1);
    splitting.resize(A->n_rows);
    f = fopen("../../../../test_data/rss_cf0", "r");
    for (int i = 0; i < A->n_rows; i++)
    {
        fscanf(f, "%d\n", &splitting[i]);
    }
    fclose(f);

    P = readMatrix("../../../../test_data/rss_P0.mtx", 0);
    P_rap = direct_interpolation(A, S, splitting);
    compare(P, P_rap);
    delete P_rap;
    delete P;

    P = readMatrix("../../../../test_data/rss_P0_mc.mtx", 0);
    P_rap = mod_classical_interpolation(A, S, splitting);
    compare(P, P_rap);
    delete P_rap;
    delete P;
    delete S;
    delete A;


    // TEST LEVEL 1
    A = readMatrix("../../../../test_data/rss_A1.mtx", 0);
    P = readMatrix("../../../../test_data/rss_P1.mtx", 0);
    S = readMatrix("../../../../test_data/rss_S1.mtx", 0);
    splitting.resize(A->n_rows);
    f = fopen("../../../../test_data/rss_cf1", "r");
    for (int i = 0; i < A->n_rows; i++)
    {
        fscanf(f, "%d\n", &splitting[i]);
    }
    fclose(f);

    P_rap = direct_interpolation(A, S, splitting);
    compare(P, P_rap);
    delete P_rap;
    delete P;

    P = readMatrix("../../../../test_data/rss_P1_mc.mtx", 0);
    P_rap = mod_classical_interpolation(A, S, splitting);
    //compare(P, P_rap);

    delete P;
    delete P_rap;
    delete S;
    delete A;
} // end of TEST(TestInterpolation, TestsInRuge_Stuben) //

