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

TEST(BiCGStabTest, TestsInKrylov)
{
    int grid[2] = {50, 50};
    double* stencil = diffusion_stencil_2d(0.001, M_PI/8.0);
    CSRMatrix* A = stencil_grid(stencil, grid, 2);
    Vector x(A->n_rows);
    Vector b(A->n_rows);
    aligned_vector<double> residuals;

    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);

    BiCGStab(A, x, b, residuals);   

    FILE* f = fopen("../../../../test_data/bicgstab_res.txt", "r");
    double res;
    for (int i = 0; i < 30; i++)
    {
        fscanf(f, "%lf\n", &res);
	    ASSERT_NEAR(res, residuals[i], 1e-06);
    }
    fclose(f);
    delete[] stencil;
    delete A;

} // end of TEST(BiCGStabTest, TestsInKrylov) //


