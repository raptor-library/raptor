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

TEST(BlockVectorRelaxationTest, TestsInUtil)
{
    int vecs_in_block = 3;
    int grid[2] = {25, 25};
    double eps = 0.001;
    double theta = M_PI/8.0;
    double* stencil = diffusion_stencil_2d(eps, theta);
    CSRMatrix* A_sten = stencil_grid(stencil, grid, 2);

    BVector x(A_sten->n_rows, vecs_in_block);
    BVector b(A_sten->n_rows, vecs_in_block);
    BVector tmp(A_sten->n_rows, vecs_in_block);

    Vector tmp_single(A_sten->n_rows);
    Vector x1(A_sten->n_rows);
    Vector x2(A_sten->n_rows);
    Vector x3(A_sten->n_rows);

    Vector b1(A_sten->n_rows);
    Vector b2(A_sten->n_rows);
    Vector b3(A_sten->n_rows);

    // Test b <- A*ones
    x1.set_rand_values(1);
    x2.set_rand_values(2);
    x3.set_rand_values(3);

    for (int i = 0; i < A_sten->n_rows; i++)
    {
        x.values[i] = x1.values[i];
        x.values[A_sten->n_rows + i] = x2.values[i];
        x.values[2*A_sten->n_rows + i] = x3.values[i];
    }

    A_sten->mult(x1, b1);
    A_sten->mult(x2, b2);
    A_sten->mult(x3, b3);

    A_sten->mult(x, b);

    // Test Jacobi
    jacobi(A_sten, x1, b1, tmp_single);
    jacobi(A_sten, x2, b2, tmp_single);
    jacobi(A_sten, x3, b3, tmp_single);
    jacobi(A_sten, x, b, tmp);
    
    for (int i = 0; i < A_sten->n_rows; i++)
    {
        ASSERT_NEAR(x.values[i], x1.values[i], 1e-06);
        ASSERT_NEAR(x.values[i + A_sten->n_rows], x2.values[i], 1e-06);
        ASSERT_NEAR(x.values[i + 2*A_sten->n_rows], x3.values[i], 1e-06);
    } 

    // Test SOR
    sor(A_sten, x1, b1, tmp_single);
    sor(A_sten, x2, b2, tmp_single);
    sor(A_sten, x3, b3, tmp_single);
    sor(A_sten, x, b, tmp);
    
    for (int i = 0; i < A_sten->n_rows; i++)
    {
        ASSERT_NEAR(x.values[i], x1.values[i], 1e-06);
        ASSERT_NEAR(x.values[i + A_sten->n_rows], x2.values[i], 1e-06);
        ASSERT_NEAR(x.values[i + 2*A_sten->n_rows], x3.values[i], 1e-06);
    } 

    // Test SSOR
    ssor(A_sten, x1, b1, tmp_single);
    ssor(A_sten, x2, b2, tmp_single);
    ssor(A_sten, x3, b3, tmp_single);
    ssor(A_sten, x, b, tmp);

    for (int i = 0; i < A_sten->n_rows; i++)
    {
        ASSERT_NEAR(x.values[i], x1.values[i], 1e-06);
        ASSERT_NEAR(x.values[i + A_sten->n_rows], x2.values[i], 1e-06);
        ASSERT_NEAR(x.values[i + 2*A_sten->n_rows], x3.values[i], 1e-06);
    } 

} // end of TEST(BlockVectorRelaxationTest, TestsInUtil) //

