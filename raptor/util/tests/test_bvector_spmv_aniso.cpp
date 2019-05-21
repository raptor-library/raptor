// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause


#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/stencil.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

} // end of main() //

TEST(AnisoBVectorSpMVTest, TestsInUtil)
{
    double b_val;
    int offset;
    int vecs_in_block = 3;
    int grid[2] = {25, 25};
    double eps = 0.001;
    double theta = M_PI/8.0;
    double* stencil = diffusion_stencil_2d(eps, theta);
    CSRMatrix* A_sten = stencil_grid(stencil, grid, 2);

    BVector *x = new BVector(A_sten->n_rows, vecs_in_block);
    BVector *b = new BVector(A_sten->n_rows, vecs_in_block);

    const char* b_ones = "../../../../test_data/aniso_ones_b.txt";
    const char* b_T_ones = "../../../../test_data/aniso_ones_b_T.txt";
    
    // Test b <- A*ones
    x->set_const_value(1.0);
    A_sten->mult(*x, *b);
    FILE* f = fopen(b_ones, "r");
    for (int i = 0; i < A_sten->n_rows; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        for (int v = 0; v < vecs_in_block; v++)
        {
            ASSERT_NEAR(b->values[i + v*b->num_values], b_val, 1e-06);
        }
    }
    fclose(f);
    
    // Test b <- A_T*ones
    A_sten->mult_T(*x, *b);
    f = fopen(b_T_ones, "r");
    for (int i = 0; i < A_sten->n_rows; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        for (int v = 0; v < vecs_in_block; v++)
        {
            ASSERT_NEAR(b->values[i + v*b->num_values], b_val, 1e-06);
        }
    } 
    fclose(f);

    delete x;
    delete b;
    delete[] stencil;
    delete A_sten;

} // end of TEST(AnisoBVectorSpMVTest, TestsInUtil) //

