// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "raptor.hpp"
#include "tests/compare.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

} // end of main() //

TEST(GaussianEliminationTest, TestsInUtil)
{
    double b_val;
    int grid[2] = {5,5};
    double eps = 0.001;
    double theta = M_PI/8.0;
    double* stencil = diffusion_stencil_2d(eps, theta);
    CSRMatrix* A_sten = stencil_grid(stencil, grid, 2);

    BVector x(A_sten->n_rows, 3);
    BVector y(A_sten->n_rows, 3);
    BVector b(A_sten->n_rows, 3);

    CSRMatrix* L = new CSRMatrix(A_sten->n_rows, A_sten->n_cols);
    CSRMatrix* U = new CSRMatrix(A_sten->n_rows, A_sten->n_cols);
    CSRMatrix* B = new CSRMatrix(A_sten->n_rows, A_sten->n_cols);

    // Test Gaussian Elimination
    A_sten->gaussian_elimination(L, U);
    B = L->spgemm(U);
    compare(A_sten, B);
    
    // Test Forward Substitution
    x.set_rand_values();
    L->mult(x, b);
    y.set_const_value(0.0);
    L->forward_substitution(y, b);
    for (int i = 0; i < x.values.size(); i++)
    {
        ASSERT_NEAR(x.values[i], y.values[i], 1e-6);
    }
    
    // Test Backward Substitution
    x.set_rand_values();
    U->mult(x, b);
    y.set_const_value(0.0);
    U->backward_substitution(y, b);
    for (int i = 0; i < x.values.size(); i++)
    {
        ASSERT_NEAR(x.values[i], y.values[i], 1e-6);
    }
   
    delete L;
    delete U;
    delete B; 

} // end of TEST(GaussianEliminationTest, TestsInUtil) //

