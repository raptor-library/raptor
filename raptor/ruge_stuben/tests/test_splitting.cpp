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
    CSRMatrix* S_py;
    CSRBoolMatrix* S;
    std::vector<int> splitting;
    std::vector<int> splitting_rap;
   
    // TEST LAPLACIAN SPLITTINGS ON LEVEL 0 
    S_py = readMatrix("../../../../test_data/rss_S0.mtx", 1);
    S = new CSRBoolMatrix(S_py);
    splitting.resize(S->n_rows);;

    // Test RugeStuben Splitting
    split_rs(S, splitting_rap);
    f = fopen("../../../../test_data/rss_cf0_rs", "r");
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
    f = fopen("../../../../test_data/weights.txt", "r");
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
    delete S_py;



    // TEST LAPLACIAN SPLITTINGS ON LEVEL 1 
    S_py = readMatrix("../../../../test_data/rss_S1.mtx", 0);
    S = new CSRBoolMatrix(S_py);
    splitting.resize(S->n_rows);;

    // Test RugeStuben Splitting
    split_rs(S, splitting_rap);
    f = fopen("../../../../test_data/rss_cf1_rs", "r");
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
    f = fopen("../../../../test_data/weights.txt", "r");
    weights.resize(S->n_rows);
    for (int i = 0; i < S->n_rows; i++)
    {
        fscanf(f, "%lf\n", &weights[i]);
    }
    fclose(f);
    split_cljp(S, splitting_rap, weights.data());
    f = fopen("../../../../test_data/rss_cf1", "r");
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
    delete S_py;

} // end of TEST(TestSplitting, TestsInRuge_Stuben) //
