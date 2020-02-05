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

    int nrhs = 5;
    int grid[2] = {50, 50};
    //int grid[2] = {5, 5};
    double* stencil = diffusion_stencil_2d(0.001, M_PI/8.0);
    ParMultilevel* ml_single;
    ParMultilevel* ml;
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);
    ParVector x(A->global_num_rows, A->local_num_rows);
    ParVector b(A->global_num_rows, A->local_num_rows);
    aligned_vector<double> residuals;
    aligned_vector<double> residuals_mincomm;
    aligned_vector<double> pre_residuals;
    
    /*ml = new ParSmoothedAggregationSolver(0.0);
    ml->setup(A, nrhs);
    ml->print_hierarchy();

    ml_single = new ParSmoothedAggregationSolver(0.0);
    ml_single->setup(A);
    ml_single->print_hierarchy();*/

    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);

    double b_norm = b.norm(2);
    EKCG(A, x, b, nrhs, residuals);
    if(rank == 0) printf("Iterations %d Final Res %e\n", residuals.size()-1, residuals[residuals.size()-1]);

    x.set_const_value(0.0);
    EKCG_MinComm(A, x, b, nrhs, residuals_mincomm);
    if(rank == 0) printf("Iterations %d Final Res %e\n", residuals_mincomm.size()-1, residuals_mincomm[residuals_mincomm.size()-1]);

    /*x.set_const_value(0.0);    
    PEKCG(ml_single, ml, A, x, b, nrhs, pre_residuals);
    if(rank == 0) printf("Iterations %d Final Res %e\n", pre_residuals.size()-1, pre_residuals[pre_residuals.size()-1]);*/

    //FILE* f = fopen("../../../../test_data/srecg_res.txt", "r");
    /*double res;
    for (int i = 0; i < residuals.size(); i++)
    {
        //fscanf(f, "%lf\n", &res);
        //ASSERT_NEAR(res, residuals[i], 1e-05);
        printf("%e\n", residuals[i]);
    }*/
    //fclose(f);

    delete[] stencil;
    delete A;
    delete ml_single;
    delete ml;
    
} // end of TEST(ParEKCGTest, TestsInKrylov) //



