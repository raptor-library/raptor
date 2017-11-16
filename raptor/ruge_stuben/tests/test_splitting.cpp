// Copyright (c) 2015-2017, RAPtor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "gallery/matrix_IO.hpp"
#include "ruge_stuben/cf_splitting.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/stencil.hpp"

#include <iostream>
#include <fstream>

using namespace raptor;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} // end of main() //


TEST(TestSplitting, TestsInRuge_Stuben)
{ 
    FILE* f;
    CSRMatrix* S;
    std::vector<int> splitting;
    std::vector<int> splitting_rap;

    const char* S0_fn = "../../../../test_data/rss_S0.pm";
    const char* S1_fn = "../../../../test_data/rss_S1.pm";
    const char* cf0_rs = "../../../../test_data/rss_cf0_rs";
    const char* cf0 = "../../../../test_data/rss_cf0";
    const char* cf1_rs = "../../../../test_data/rss_cf1_rs";
    const char* cf1 = "../../../../test_data/rss_cf1";
    const char* weights_fn = "../../../../test_data/weights.txt";
   
    // TEST LAPLACIAN SPLITTINGS ON LEVEL 0 
    S = readMatrix(S0_fn);
    splitting.resize(S->n_rows);;

    // Test RugeStuben Splitting
    split_rs(S, splitting_rap);
    f = fopen(cf0_rs, "r");
    for (int i = 0; i < S->n_rows; i++)
    {
        fscanf(f, "%d\n", &splitting[i]);
    }
    fclose(f);
    ASSERT_EQ(splitting_rap.size(), splitting.size());
    for (int i = 0; i < S->n_rows; i++)
    {
        ASSERT_EQ(splitting[i], splitting_rap[i]);
    }


    // Test CLJP Splittings
    f = fopen(weights_fn, "r");
    std::vector<double> weights(S->n_rows);
    for (int i = 0; i < S->n_rows; i++)
    {
        fscanf(f, "%lf\n", &weights[i]);
    }
    fclose(f);
    split_cljp(S, splitting_rap, weights.data());
    f = fopen("../../../../test_data/rss_cf0", "r");
    for (int i = 0; i < S->n_rows; i++)
    {
        fscanf(f, "%d\n", &splitting[i]);
    }
    fclose(f);
    ASSERT_EQ(splitting_rap.size(), splitting.size());
    for (int i = 0; i < S->n_rows; i++)
    {
        ASSERT_EQ(splitting[i], splitting_rap[i]);
    }

    delete S;



    // TEST LAPLACIAN SPLITTINGS ON LEVEL 1 
    S = readMatrix(S1_fn);
    splitting.resize(S->n_rows);;

    // Test RugeStuben Splitting
    split_rs(S, splitting_rap);
    f = fopen(cf1_rs, "r");
    for (int i = 0; i < S->n_rows; i++)
    {
        fscanf(f, "%d\n", &splitting[i]);
    }
    fclose(f);
    ASSERT_EQ(splitting_rap.size(), splitting.size());
    for (int i = 0; i < S->n_rows; i++)
    {
        ASSERT_EQ(splitting[i], splitting_rap[i]);
    }


    // Test CLJP Splittings
    f = fopen(weights_fn, "r");
    weights.resize(S->n_rows);
    for (int i = 0; i < S->n_rows; i++)
    {
        fscanf(f, "%lf\n", &weights[i]);
    }
    fclose(f);
    split_cljp(S, splitting_rap, weights.data());
    f = fopen(cf1, "r");
    for (int i = 0; i < S->n_rows; i++)
    {
        fscanf(f, "%d\n", &splitting[i]);
    }
    fclose(f);
    ASSERT_EQ(splitting_rap.size(), splitting.size());
    for (int i = 0; i < S->n_rows; i++)
    {
        ASSERT_EQ(splitting[i], splitting_rap[i]);
    }

    delete S;

} // end of TEST(TestSplitting, TestsInRuge_Stuben) //
