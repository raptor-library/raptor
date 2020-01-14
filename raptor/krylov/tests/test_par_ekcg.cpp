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

TEST(ParEKCGTest, TestsInKrylov)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    //int grid[2] = {50, 50};
    int grid[2] = {5, 5};
    double* stencil = diffusion_stencil_2d(0.001, M_PI/8.0);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);
    ParVector x(A->global_num_rows, A->local_num_rows);
    ParVector b(A->global_num_rows, A->local_num_rows);
    aligned_vector<double> residuals;

    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);

    double b_norm = b.norm(2);
    EKCG(A, x, b, 2, residuals);

    //FILE* f = fopen("../../../../test_data/srecg_res.txt", "r");
    double res;
    for (int i = 0; i < residuals.size(); i++)
    {
        //fscanf(f, "%lf\n", &res);
        //ASSERT_NEAR(res, residuals[i], 1e-05);
        printf("%e\n", residuals[i]);
    }
    //fclose(f);

    delete[] stencil;
    delete A;
    
} // end of TEST(ParEKCGTest, TestsInKrylov) //



