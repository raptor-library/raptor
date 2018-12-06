// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "aorthonormalization/cgs.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/stencil.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

} // end of main() //

TEST(CGSTest, TestsInUtil)
{
    int num_vectors = 4;
    int grid[2] = {50, 50};
    double* stencil = diffusion_stencil_2d(0.001, M_PI/8.0);
    CSRMatrix* A = stencil_grid(stencil, grid, 2);
    
    int Q_bvecs = 5;
    int W_bvecs = 2;
    int len = 20;
    double val;

    BVector *Q = new BVector(len, Q_bvecs);
    BVector *W = new BVector(len, W_bvecs);

    Q->set_const_value(1.0);
    W->set_const_value(1.0);
    std::vector<double> alphas = {1.0, 2.0};
    W->scale(1.0, &(alphas[0]));

    delete A;
    delete[] stencil;
    delete Q; 
    delete W;

} // end of TEST(CGSTest, TestsInUtil) //

