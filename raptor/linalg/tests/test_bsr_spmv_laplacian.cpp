// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "gtest/gtest.h"
#include "raptor.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

} // end of main() //

TEST(LaplacianBSRSpMVTest, TestsInUtil)
{
/*    double b_val;
    int grid[3] = {10, 10, 10};
    double* stencil = laplace_stencil_27pt();
    const CSRMatrix* A_sten = stencil_grid(stencil, grid, 3);
    BSRMatrix* B_sten = new BSRMatrix(A_sten, 2, 2);

    Vector x(B_sten->n_rows);
    Vector b(B_sten->n_rows);
    
    // Test b <- A*ones
    x.set_const_value(1.0);
    B_sten->mult(x, b);
    FILE* f = fopen("../../../../test_data/laplacian27_ones_b.txt", "r");
    for (int i = 0; i < B_sten->n_rows; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        ASSERT_NEAR(b[i], b_val, 1e-06);
    } 
    fclose(f);

    // Test b <- A_T*ones
    B_sten->mult_T(x, b);
    f = fopen("../../../../test_data/laplacian27_ones_b_T.txt", "r");
    for (int i = 0; i < B_sten->n_rows; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        ASSERT_NEAR(b[i], b_val, 1e-06);
    } 
    fclose(f);

    // Tests b <- A*incr
    for (int i = 0; i < B_sten->n_rows; i++)
    {
        x[i] = i;
    }
    B_sten->mult(x, b);
    f = fopen("../../../../test_data/laplacian27_inc_b.txt", "r");
    for (int i = 0; i < B_sten->n_rows; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        ASSERT_NEAR(b[i], b_val, 1e-06);
    } 
    fclose(f);

    // Tests b <- A_T*incr
    B_sten->mult_T(x, b);
    f = fopen("../../../../test_data/laplacian27_inc_b_T.txt", "r");
    for (int i = 0; i < B_sten->n_rows; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        ASSERT_NEAR(b[i], b_val, 1e-06);
    } 
    fclose(f);

    delete[] stencil;
    delete A_sten;
    delete B_sten;*/
} // end of TEST(LaplacianSpMVTest, TestsInUtil) //

