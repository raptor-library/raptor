// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "gtest/gtest.h"
#include "raptor/raptor.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

} // end of main() //

TEST(LaplacianSpMVTest, TestsInUtil)
{
    double b_val;
    int grid[3] = {10, 10, 10};
    double* stencil = laplace_stencil_27pt();
    CSRMatrix* A_sten = stencil_grid(stencil, grid, 3);

    Vector x(A_sten->n_rows);
    Vector b(A_sten->n_rows);
    
    BSRMatrix* A = new BSRMatrix(A_sten, 5, 5);

    // Test b <- A*ones
    int n_items_read;
    x.set_const_value(1.0);
    A->mult(x, b);
    FILE* f = fopen("../../../../test_data/laplacian27_ones_b.txt", "r");
    for (int i = 0; i < A_sten->n_rows; i++)
    {
        n_items_read = fscanf(f, "%lg\n", &b_val);
        ASSERT_EQ(n_items_read, 1);
        ASSERT_NEAR(b[i], b_val, 1e-06);
    } 
    fclose(f);

    // Test b <- A_T*ones
    A->mult_T(x, b);
    f = fopen("../../../../test_data/laplacian27_ones_b_T.txt", "r");
    for (int i = 0; i < A_sten->n_rows; i++)
    {
        n_items_read = fscanf(f, "%lg\n", &b_val);
        ASSERT_EQ(n_items_read, 1);
        ASSERT_NEAR(b[i], b_val, 1e-06);
    } 
    fclose(f);

    // Tests b <- A*incr
    for (int i = 0; i < A_sten->n_rows; i++)
    {
        x[i] = i;
    }
    A->mult(x, b);
    f = fopen("../../../../test_data/laplacian27_inc_b.txt", "r");
    for (int i = 0; i < A_sten->n_rows; i++)
    {
        n_items_read = fscanf(f, "%lg\n", &b_val);
        ASSERT_EQ(n_items_read, 1);
        ASSERT_NEAR(b[i], b_val, 1e-06);
    } 
    fclose(f);

    // Tests b <- A_T*incr
    A->mult_T(x, b);
    f = fopen("../../../../test_data/laplacian27_inc_b_T.txt", "r");
    for (int i = 0; i < A_sten->n_rows; i++)
    {
        n_items_read = fscanf(f, "%lg\n", &b_val);
        ASSERT_EQ(n_items_read, 1);
        ASSERT_NEAR(b[i], b_val, 1e-06);
    } 
    fclose(f);


    delete A;
    delete A_sten;

} // end of TEST(LaplacianSpMVTest, TestsInUtil) //

