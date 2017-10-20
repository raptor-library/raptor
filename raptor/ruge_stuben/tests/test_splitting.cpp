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
    CSRMatrix* S;
    std::vector<int> splitting;
    std::vector<int> splitting_rap;
   
    // TEST LAPLACIAN SPLITTING ON LEVEL 0 //
    S = readMatrix("../../tests/rss_laplace_S0.mtx", 1);
    // Read pyamg splitting
    splitting.resize(S->n_rows);;
    f = fopen("../../tests/rss_laplace_cf0.txt", "r");
    for (int i = 0; i < S->n_rows; i++)
    {
        fscanf(f, "%d\n", &splitting[i]);
    }
    fclose(f);
    // Create cf splitting
    split_rs(S, splitting_rap);
    //Compare splittings
    ASSERT_EQ(splitting_rap.size(), splitting.size());
    for (int i = 0; i < S->n_rows; i++)
    {
        ASSERT_EQ(splitting[i], splitting_rap[i]);
    }
    delete S;

    // TEST LAPLACIAN SPLITTING ON LEVEL 1 //
    S = readMatrix("../../tests/rss_laplace_S1.mtx", 0);
    // Read pyamg splitting
    splitting.resize(S->n_rows);;
    f = fopen("../../tests/rss_laplace_cf1.txt", "r");
    for (int i = 0; i < S->n_rows; i++)
    {
        fscanf(f, "%d\n", &splitting[i]);
    }
    fclose(f);
    // Create cf splitting
    split_rs(S, splitting_rap);
    //Compare splittings
    ASSERT_EQ(splitting_rap.size(), splitting.size());
    for (int i = 0; i < S->n_rows; i++)
    {
        ASSERT_EQ(splitting[i], splitting_rap[i]);
    }
    delete S;

    // TEST ANISO SPLITTING ON LEVEL 0 //
    S = readMatrix("../../tests/rss_aniso_S0.mtx", 1);
    // Read pyamg splitting
    splitting.resize(S->n_rows);;
    f = fopen("../../tests/rss_aniso_cf0.txt", "r");
    for (int i = 0; i < S->n_rows; i++)
    {
        fscanf(f, "%d\n", &splitting[i]);
    }
    fclose(f);
    // Create cf splitting
    split_rs(S, splitting_rap);
    //Compare splittings
    ASSERT_EQ(splitting_rap.size(), splitting.size());
    for (int i = 0; i < S->n_rows; i++)
    {
        ASSERT_EQ(splitting[i], splitting_rap[i]);
    }
    delete S;

    // TEST ANISO SPLITTING ON LEVEL 1 //
    S = readMatrix("../../tests/rss_aniso_S1.mtx", 0);
    // Read pyamg splitting
    splitting.resize(S->n_rows);;
    f = fopen("../../tests/rss_aniso_cf1.txt", "r");
    for (int i = 0; i < S->n_rows; i++)
    {
        fscanf(f, "%d\n", &splitting[i]);
    }
    fclose(f);
    // Create cf splitting
    split_rs(S, splitting_rap);
    //Compare splittings
    ASSERT_EQ(splitting_rap.size(), splitting.size());
    for (int i = 0; i < S->n_rows; i++)
    {
        ASSERT_EQ(splitting[i], splitting_rap[i]);
    }
    delete S;

    // TEST 10 x 10 2D rotated aniso... print this one for graphing //
    int grid[2] = {10, 10};
    double eps = 0.001;
    double theta = M_PI/8.0;
    double* stencil = diffusion_stencil_2d(eps, theta);
    CSRMatrix* A = stencil_grid(stencil, grid, 2);
    S = A->strength();
    split_rs(S, splitting_rap);
    ofstream outfile;
    outfile.open("aniso_splitting.txt");
    for (int i = 0; i < A->n_rows; i++)
    {
        outfile << splitting_rap[i] << endl;
    }
    outfile.close();

} // end of TEST(TestSplitting, TestsInRuge_Stuben) //

