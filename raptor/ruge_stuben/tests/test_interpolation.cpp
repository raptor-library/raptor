// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

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
    aligned_vector<int> splitting;
    FILE* f;

    const char* A0_fn = "../../../../test_data/rss_A0.pm";
    const char* A1_fn = "../../../../test_data/rss_A1.pm";
    const char* S0_fn = "../../../../test_data/rss_S0.pm";
    const char* S1_fn = "../../../../test_data/rss_S1.pm";
    const char* cf0_fn = "../../../../test_data/rss_cf0.txt";
    const char* cf1_fn = "../../../../test_data/rss_cf1.txt";
    const char* P0_fn = "../../../../test_data/rss_P0.pm";
    const char* P0_mc_fn = "../../../../test_data/rss_P0_mc.pm";
    const char* P0_extend = "../../../../test_data/rss_P0_extend.pm";
    const char* P1_fn = "../../../../test_data/rss_P1.pm";
    const char* P1_mc_fn = "../../../../test_data/rss_P1_mc.pm";
    const char* P1_extend = "../../../../test_data/rss_P1_extend.pm";

    // TEST LEVEL 0
    A = readMatrix(A0_fn);
    S = readMatrix(S0_fn);
    splitting.resize(A->n_rows);
    f = fopen(cf0_fn, "r");
    for (int i = 0; i < A->n_rows; i++)
    {
        fscanf(f, "%d\n", &splitting[i]);
    }
    fclose(f);

    P = readMatrix(P0_fn);
    P_rap = direct_interpolation(A, S, splitting);
    compare(P, P_rap);
    delete P_rap;
    delete P;

    P = readMatrix(P0_mc_fn);
    P_rap = mod_classical_interpolation(A, S, splitting);
    compare(P, P_rap);
    delete P_rap;
    delete P;

    P = readMatrix(P0_extend);
    P_rap = extended_interpolation(A, S, splitting);
    compare(P, P_rap);
    delete P_rap;
    delete P;

    delete S;
    delete A;

    // TEST LEVEL 1
    A = readMatrix(A1_fn);
    S = readMatrix(S1_fn);
    splitting.resize(A->n_rows);
    f = fopen(cf1_fn, "r");
    for (int i = 0; i < A->n_rows; i++)
    {
        fscanf(f, "%d\n", &splitting[i]);
    }
    fclose(f);

    P = readMatrix(P1_fn);
    P_rap = direct_interpolation(A, S, splitting);
    compare(P, P_rap);
    delete P_rap;
    delete P;

    // TODO -- serial mod classical interp not working
    P = readMatrix(P1_mc_fn);
    P_rap = mod_classical_interpolation(A, S, splitting);
    compare(P, P_rap);

    delete P;
    delete P_rap;

    P = readMatrix(P1_extend);
    P_rap = extended_interpolation(A, S, splitting);
    compare(P, P_rap);
    delete P_rap;
    delete P;

    delete S;
    delete A;
} // end of TEST(TestInterpolation, TestsInRuge_Stuben) //

