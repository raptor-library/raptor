// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause


#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/stencil.hpp"
#include "mpi.h"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp = RUN_ALL_TESTS();
    MPI_Finalize();

    return temp;

} // end of main() //

TEST(AnisoSpMVTest, TestsInUtil)
{
    double start, end;

    double b_val;
    int grid[2] = {15, 15};
    double eps = 0.001;
    double theta = M_PI/8.0;
    double* stencil = diffusion_stencil_2d(eps, theta);
    CSRMatrix* A_sten = stencil_grid(stencil, grid, 2);
    CSRMatrix* B;
    CSRMatrix* C;

    int val[2];
    int iters = 50;
    start = MPI_Wtime();
    for (int i = 0; i < iters; i++)
    {
        CSCMatrix* A_csc;
        A_csc = A_sten->to_CSC(); 
        B = A_sten->spgemm_T(A_csc);
        delete A_csc;
    }
    end = MPI_Wtime();
    printf("spgemm_T time %f\n", (end-start)/iters);
    
    start = MPI_Wtime();
    for (int i = 0; i < iters; i++)
    {
        C = A_sten->spgemm(A_sten);
    }
    end = MPI_Wtime();
    printf("spgemm time %f\n", (end-start)/iters);

    delete A_sten;
    delete B;
    delete C;

} // end of TEST(AnisoSpMVTest, TestsInUtil) //

