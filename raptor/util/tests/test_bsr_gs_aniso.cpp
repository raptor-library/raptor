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

TEST(AnisoJacobiTest, TestsInUtil)
{
    double x_val;
    int grid[2] = {25, 25};
    double eps = 0.001;
    double theta = M_PI/8.0;
    double* stencil = diffusion_stencil_2d(eps, theta);
    CSRMatrix* A_sten = stencil_grid(stencil, grid, 2);
    int n_items_read;
    FILE* f;

    Vector x(A_sten->n_rows);
    Vector b(A_sten->n_rows);
    Vector tmp(A_sten->n_rows);

    BSRMatrix* A = new BSRMatrix(A_sten, 5, 5);
    double* D_inv;

    const char* x_ones_1 = "../../../../test_data/aniso_block_gs_ones_1.txt";
    const char* x_ones_2 = "../../../../test_data/aniso_block_gs_ones_2.txt";
    const char* x_inc_1 = "../../../../test_data/aniso_block_gs_inc_1.txt";
    const char* x_inc_2 = "../../../../test_data/aniso_block_gs_inc_2.txt";

    // SETUP D_inv!
    block_relax_init(A, &D_inv);
    
    /*********************************************
     *  Test jacobi when b is constant value 1.0 
     *********************************************/
    // Iteration 1
    b.set_const_value(1.0);
    x.set_const_value(0.0);
    sor(A, D_inv, b, x, tmp, 1);
    f = fopen(x_ones_1, "r");
    for (int i = 0; i < A_sten->n_rows; i++)
    {
        n_items_read = fscanf(f, "%lg\n", &x_val);
        ASSERT_EQ(n_items_read, 1);
        ASSERT_NEAR(x[i],x_val,1e-06);
    } 
    fclose(f);

    // Iteration 2
    sor(A, D_inv, b, x, tmp, 1);
    f = fopen(x_ones_2, "r");
    for (int i = 0; i < A_sten->n_rows; i++)
    {
        n_items_read = fscanf(f, "%lg\n", &x_val);
        ASSERT_EQ(n_items_read, 1);
        ASSERT_NEAR(x[i],x_val,1e-06);
    } 
    fclose(f);
    

    /*********************************************
     *  Test jacobi when b[i] = i
     *********************************************/
    // Iteration 1
    for (int i = 0; i < A_sten->n_rows; i++)
    {
        b[i] = i;
    }
    x.set_const_value(0.0);
    sor(A, D_inv, b, x, tmp, 1);
    f = fopen(x_inc_1, "r");
    for (int i = 0; i < A_sten->n_rows; i++)
    {
        n_items_read = fscanf(f, "%lg\n", &x_val);
        ASSERT_EQ(n_items_read, 1);
        ASSERT_NEAR(x[i],x_val,1e-06);
    } 
    fclose(f);

    // Iteration 2
    sor(A, D_inv, b, x, tmp, 1);
    f = fopen(x_inc_2, "r");
    for (int i = 0; i < A_sten->n_rows; i++)
    {
        n_items_read = fscanf(f, "%lg\n", &x_val);
        ASSERT_EQ(n_items_read, 1);
        ASSERT_NEAR(x[i],x_val,1e-06);
    } 
    fclose(f);

    // Free D_inv
    block_relax_free(D_inv);

    delete A;
    delete A_sten;

} // end of TEST(AnisoSpMVTest, TestsInUtil) //

