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

TEST(LaplacianTest, TestsInGallery)
{
    int start, end;

    int grid[3] = {10, 10, 10};
    double* stencil = laplace_stencil_27pt();
    CSRMatrix* A_sten = stencil_grid(stencil, grid, 3);

    const char* mat_fn = "../../../../test_data/laplacian.pm";
    CSRMatrix* A_io = readMatrix(mat_fn);

    // Compare shapes
    ASSERT_EQ(A_io->n_rows, A_sten->n_rows);
    ASSERT_EQ(A_io->n_cols, A_sten->n_cols);

    A_sten->sort();
    A_io->sort();

    ASSERT_EQ(A_sten->idx1[0], A_io->idx1[0]);

    for (int i = 0; i < A_io->n_rows; i++)
    {
        // Check correct row_ptrs
        ASSERT_EQ(A_sten->idx1[i+1], A_io->idx1[i+1]);
        start = A_sten->idx1[i];
        end   = A_sten->idx1[i+1];

        // Check correct col indices / values
        for (int j = start; j < end; j++)
        {
            ASSERT_EQ(A_sten->idx2[j], A_io->idx2[j]);
            ASSERT_NEAR(A_sten->vals[j], A_io->vals[j], zero_tol);
        }
    }

    delete[] stencil;
    delete A_sten;
    delete A_io;
} // end of TEST(LaplacianTest, TestsInGallery) //

