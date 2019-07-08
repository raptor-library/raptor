// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "multilevel/par_multilevel.hpp"
#include "ruge_stuben/par_ruge_stuben_solver.hpp"
#include "aggregation/par_smoothed_aggregation_solver.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/laplacian27pt.hpp"
#include "gallery/par_stencil.hpp"

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
    
    int dim = 3;
    int nrhs = 3;

    int grid[3] = {5, 5, 5};

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
    x.resize(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    b.resize(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    
    ml = new ParRugeStubenSolver(strong_threshold, CLJP, ModClassical, Classical, SOR);
    ml->setup(A, nrhs);
    
    x_single.resize(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    b_single.resize(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    
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
    delete ml_single;

    // Test Smoothed Aggregation Solver
    ml = new ParSmoothedAggregationSolver(strong_threshold, MIS, JacobiProlongation, Symmetric, SOR);
    ml->setup(A, nrhs);
    
    ml_single = new ParSmoothedAggregationSolver(strong_threshold, MIS, JacobiProlongation, Symmetric, SOR);
    ml_single->setup(A);
    
    x.set_const_value(1.0);
    x.scale(1.0, &(alphas[0]));

    A->mult(x, b);
    x.set_const_value(0.0);
    iter = ml->solve(x, b);
    
    // Check residuals
    aligned_vector<double>& mrhs_sa = ml->get_residuals();
    x_single.set_const_value(1.0);
    A->mult(x_single, b_single);
    x_single.set_const_value(0.0);
    iter = ml_single->solve(x_single, b_single);
    aligned_vector<double>& single_sa = ml_single->get_residuals();
    for (int i = 0; i < iter; i++)
    {
        for (int v = 0; v < nrhs; v++)
        {
            ASSERT_NEAR(mrhs_sa[i*nrhs + v], single_sa[i], 1e-10);
        }
    }

    delete ml;
    delete ml_single;

    delete A;

} // end of TEST(ParBlockAMGTest, TestsInMultilevel) //
