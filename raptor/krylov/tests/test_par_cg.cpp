// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "raptor.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp=RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //

TEST(ParCGTest, TestsInKrylov)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int grid[2] = {50, 50};
    double* stencil = diffusion_stencil_2d(0.001, M_PI/8.0);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);
    ParVector x(A->global_num_rows, A->local_num_rows);
    ParVector b(A->global_num_rows, A->local_num_rows);
    std::vector<double> residuals;
    int n_items_read;

    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);

    double b_norm = b.norm(2);
    CG(A, x, b, residuals);

    FILE* f = fopen("../../../../test_data/cg_res.txt", "r");
    double res;
    for (int i = 0; i < (int) residuals.size(); i++)
    {
        n_items_read = fscanf(f, "%lf\n", &res);
        ASSERT_EQ(n_items_read, 1);
        ASSERT_NEAR(res, residuals[i] * b_norm, 1e-06);
    }
    fclose(f);


    delete[] stencil;
    delete A;
    
} // end of TEST(ParCGTest, TestsInKrylov) //



