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


TEST(ParTAPAMGTest, TestsInMultilevel)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    setenv("PPN", "4", 1);
    
    int dim = 2;
    //int grid[2] = {25, 25};
    int grid[2] = {10, 10};
    double eps = 0.001;
    double theta = M_PI / 8.0;

    ParMultilevel* ml;
    ParCSRMatrix* A;
    ParVector x;
    ParVector b;

    double strong_threshold = 0.0;

    double* stencil = diffusion_stencil_2d(eps, theta);
    A = par_stencil_grid(stencil, grid, dim);
    delete[] stencil;

    x.resize(A->global_num_rows, A->local_num_rows);
    b.resize(A->global_num_rows, A->local_num_rows);

    int iter;
    
    /*if (rank == 0) printf("******* 3-step *******\n");
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
   
    // 3-step TAPComm Test // 
    ml = new ParRugeStubenSolver(strong_threshold, CLJP, ModClassical, Classical, SOR);
    ml->tap_amg = 0;
    ml->setup(A);
    ml->print_hierarchy();

    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);
    iter = ml->solve(x, b);
    ml->print_residuals(iter);

    delete ml;*/

    if (rank == 0) printf("******* 2-step *******\n");
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // 2-step TAPComm Test //
    ml = new ParRugeStubenSolver(strong_threshold, CLJP, ModClassical, Classical, SOR);
    ml->tap_amg = 0;
    ml->tap_simple = true;
    ml->setup(A);
    ml->print_hierarchy();

    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);
    iter = ml->solve(x, b);
    ml->print_residuals(iter);

    delete ml;
    delete A;

} // end of TEST(ParTAPAMGTest, TestsInMultilevel) //
