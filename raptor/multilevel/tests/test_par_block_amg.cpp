// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "raptor.hpp"

using namespace raptor;


int argc;
char **argv;

int main(int _argc, char** _argv)
{
    MPI_Init(&_argc, &_argv);
    
    ::testing::InitGoogleTest(&_argc, _argv);
    argc = _argc;
    argv = _argv;
    int temp=RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //


TEST(ParBlockAMGTest, TestsInMultilevel)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    setenv("PPN", "4", 1);
    
    int dim = 3;
    int nrhs = 3;

    int grid[3] = {10, 10, 10};

    ParMultilevel* ml;
    ParCSRMatrix* A;
    ParBVector x;
    ParBVector b;
    
    ParMultilevel* ml_single;
    ParVector x_single;
    ParVector b_single;

    double strong_threshold = 0.0;

    double* stencil = laplace_stencil_27pt();
    A = par_stencil_grid(stencil, grid, dim);
    delete[] stencil;

    x.local->b_vecs = nrhs;
    b.local->b_vecs = nrhs;
    x.resize(A->global_num_rows, A->local_num_rows);
    b.resize(A->global_num_rows, A->local_num_rows);
    
    /*ml = new ParRugeStubenSolver(strong_threshold, CLJP, ModClassical, Classical, SOR);
    ml->setup(A, nrhs);
    ml->print_hierarchy();

    x_single.resize(A->global_num_rows, A->local_num_rows);
    b_single.resize(A->global_num_rows, A->local_num_rows);
    
    ml_single = new ParRugeStubenSolver(strong_threshold, CLJP, ModClassical, Classical, SOR);
    ml_single->setup(A);

    x.set_const_value(1.0);
    std::vector<double> alphas = {1.0, 2.0, 3.0};
    x.scale(1.0, &(alphas[0]));

    A->mult(x, b);
    x.set_const_value(0.0);
    int iter = ml->solve(x, b);

    // Check residuals
    aligned_vector<double>& mrhs_res = ml->get_residuals();
    x_single.set_const_value(1.0);
    A->mult(x_single, b_single);
    x_single.set_const_value(0.0);
    iter = ml_single->solve(x_single, b_single);
    aligned_vector<double>& single_res = ml_single->get_residuals();
    for (int i = 0; i < iter; i++)
    {
        for (int v = 0; v < nrhs; v++)
        {
            ASSERT_NEAR(mrhs_res[i*nrhs + v], single_res[i], 1e-10);
        }
    }

    delete ml;
    delete ml_single;*/
   
    // Test Standard TAP AMG with block vectors
    /*ml = new ParRugeStubenSolver(strong_threshold, CLJP, ModClassical, Classical, SOR);
    ml->tap_amg = 0;
    ml->max_iterations = 3;
    ml->setup(A, nrhs);

    ml_single = new ParRugeStubenSolver(strong_threshold, CLJP, ModClassical, Classical, SOR);
    ml_single->tap_amg = 0;
    ml->max_iterations = 3;
    ml_single->setup(A);

    x.set_const_value(1.0);
    std::vector<double> alphas = {1.0, 2.0, 3.0};
    x.scale(1.0, &(alphas[0]));

    A->mult(x, b);
    x.set_const_value(0.0);
    ml->cycle(x, b);*/

    MPI_Barrier(MPI_COMM_WORLD);
    A->mult(x_single, b_single);
    //x_single.set_const_value(0.0);
    //ml_single->cycle(x_single, b_single);

    //int iter = ml->solve(x, b);

    // Check residuals
    /*aligned_vector<double>& mrhs_tap_res = ml->get_residuals();
    x_single.set_const_value(1.0);
    A->mult(x_single, b_single);
    x_single.set_const_value(0.0);
    iter = ml_single->solve(x_single, b_single);
    aligned_vector<double>& single_tap_res = ml_single->get_residuals();
    for (int i = 0; i < iter; i++)
    {
        for (int v = 0; v < nrhs; v++)
        {
            //ASSERT_NEAR(mrhs_tap_res[i*nrhs + v], single_tap_res[i], 1e-10);
            //printf("res[%d] %e mrhs[%d] %e\n", i, single_tap_res[i], i*nrhs+v, mrhs_tap_res[i*nrhs + v]);
        }
    }*/

    delete ml;
    delete ml_single;
    
    // Test Simple TAP AMG with block vectors
    /*ml = new ParRugeStubenSolver(strong_threshold, CLJP, ModClassical, Classical, SOR);
    ml->tap_amg = 0;
    ml->setup(A, nrhs);

    ml_single = new ParRugeStubenSolver(strong_threshold, CLJP, ModClassical, Classical, SOR);
    ml_single->setup(A);

    x.set_const_value(1.0);
    x.scale(1.0, &(alphas[0]));

    A->mult(x, b);
    x.set_const_value(0.0);
    iter = ml->solve(x, b);

    // Check residuals
    aligned_vector<double>& mrhs_tap_res = ml->get_residuals();
    x_single.set_const_value(1.0);
    A->mult(x_single, b_single);
    x_single.set_const_value(0.0);
    iter = ml_single->solve(x_single, b_single);
    aligned_vector<double>& single_tap_res = ml_single->get_residuals();
    for (int i = 0; i < iter; i++)
    {
        for (int v = 0; v < nrhs; v++)
        {
            ASSERT_NEAR(mrhs_tap_res[i*nrhs + v], single_tap_res[i], 1e-10);
            //printf("res[%d] %e mrhs[%d] %e\n", i, single_tap_res[i], i*nrhs+v, mrhs_tap_res[i*nrhs + v]);
        }
    }*/



    // Test Smoothed Aggregation Solver
    /*ml = new ParSmoothedAggregationSolver(strong_threshold);
    ml->setup(A);

    if (rank == 0)
    {
        printf("Num Levels = %d\n", ml->num_levels);
	    printf("A\tNRow\tNCol\tNNZ\n");
    }
    for (int i = 0; i < ml->num_levels; i++)
    {
        ParCSRMatrix* Al = ml->levels[i]->A;
	    long lcl_nnz = Al->local_nnz;
	    long nnz;
	    MPI_Reduce(&lcl_nnz, &nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	    if (rank == 0)
	    {
            printf("%d\t%d\t%d\t%lu\n", i, Al->global_num_rows, Al->global_num_cols, nnz);
        }
    }

    if (rank == 0)
    {
	printf("\nP\tNRow\tNCol\tNNZ\n");
    }
    for (int i = 0; i < ml->num_levels-1; i++)
    {
        ParCSRMatrix* Pl = ml->levels[i]->P;
	    long lcl_nnz = Pl->local_nnz;
	    long nnz;
	    MPI_Reduce(&lcl_nnz, &nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	    if (rank == 0)
	    {
            printf("%d\t%d\t%d\t%lu\n", i, Pl->global_num_rows, Pl->global_num_cols, nnz);
	    }
    }
    
    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);
    iter = ml->solve(x, b);
    if (rank == 0)
    {
        printf("\nSolve Phase Relative Residuals:\n");
    }
    aligned_vector<double>& sa_res = ml->get_residuals();
    if (rank == 0) for (int i = 0; i < iter; i++)
    {
        printf("Res[%d] = %e\n", i, sa_res[i]);
    }

    delete ml;
    delete ml_single;*/

    delete A;
    
    setenv("PPN", "16", 1);

} // end of TEST(ParAMGTest, TestsInMultilevel) //
