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


TEST(ParBlockSmoothAggTest, TestsInMultilevel)
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

    x_single.resize(A->global_num_rows, A->local_num_rows);
    b_single.resize(A->global_num_rows, A->local_num_rows);
    
    ml = new ParSmoothedAggregationSolver(strong_threshold);
    ml->setup(A, nrhs);
    ml->print_hierarchy();
    
    ml_single = new ParSmoothedAggregationSolver(strong_threshold);
    ml_single->setup(A);
    ml->print_hierarchy();

    x.set_const_value(1.0);
    std::vector<double> alphas = {1.0, 2.0, 3.0};
    x.scale(1.0, &(alphas[0]));

    A->mult(x, b);
    x.set_const_value(0.0);
    ml->cycle(x, b);

    x_single.set_const_value(1.0); 
    A->mult(x_single, b_single);
    x_single.set_const_value(0.0);
    ml_single->cycle(x_single, b_single);

    // Compare values in x_single and x
    for (int i = 0; i < x.local_n; i++)
    {
        ASSERT_NEAR(x.local->values[i], x_single.local->values[i], 1e-10);
    }
   
    x_single.set_const_value(2.0); 
    A->mult(x_single, b_single);
    x_single.set_const_value(0.0);
    ml_single->cycle(x_single, b_single);
    
    // Compare values in x_single and x
    for (int i = 0; i < x.local_n; i++)
    {
        ASSERT_NEAR(x.local->values[x.local_n + i], x_single.local->values[i], 1e-10);
    }
    
    x_single.set_const_value(3.0); 
    A->mult(x_single, b_single);
    x_single.set_const_value(0.0);
    ml_single->cycle(x_single, b_single);
    
    // Compare values in x_single and x
    for (int i = 0; i < x.local_n; i++)
    {
        ASSERT_NEAR(x.local->values[2*x.local_n + i], x_single.local->values[i], 1e-10);
    }

    x.set_const_value(0.0);
    x_single.set_const_value(0.0);

    int iter_block = ml->solve(x, b);
    int iter_single = ml_single->solve(x_single, b_single);

    // Check residuals
    aligned_vector<double>& mrhs_res = ml->get_residuals();
    aligned_vector<double>& single_res = ml_single->get_residuals();
    for (int i = 0; i < iter_single; i++)
    {
        for (int v = 0; v < nrhs; v++)
        {
            ASSERT_NEAR(mrhs_res[i*nrhs + v], single_res[i], 1e-10);
        }
    }

    delete ml;
    delete ml_single;
    
    // Test 3-step TAP AMG with block vectors
    ml = new ParSmoothedAggregationSolver(strong_threshold);
    ml->tap_amg = 0;
    ml->setup(A, nrhs);

    ml_single = new ParSmoothedAggregationSolver(strong_threshold);
    ml_single->tap_amg = 0;
    ml_single->setup(A);
    
    x.set_const_value(1.0);
    x.scale(1.0, &(alphas[0]));

    A->mult(x, b);
    x.set_const_value(0.0);
    ml->cycle(x, b);

    x_single.set_const_value(1.0); 
    A->mult(x_single, b_single);
    x_single.set_const_value(0.0);
    ml_single->cycle(x_single, b_single);
    
    // Compare values in x_single and x
    for (int i = 0; i < x.local_n; i++)
    {
        ASSERT_NEAR(x.local->values[i], x_single.local->values[i], 1e-10);
    }
   
    x_single.set_const_value(2.0); 
    A->mult(x_single, b_single);
    x_single.set_const_value(0.0);
    ml_single->cycle(x_single, b_single);
    
    // Compare values in x_single and x
    for (int i = 0; i < x.local_n; i++)
    {
        ASSERT_NEAR(x.local->values[x.local_n + i], x_single.local->values[i], 1e-10);
    }
    
    x_single.set_const_value(3.0); 
    A->mult(x_single, b_single);
    x_single.set_const_value(0.0);
    ml_single->cycle(x_single, b_single);
    
    // Compare values in x_single and x
    for (int i = 0; i < x.local_n; i++)
    {
        ASSERT_NEAR(x.local->values[2*x.local_n + i], x_single.local->values[i], 1e-10);
    }

    delete ml;
    delete ml_single;
    
    // Test 2-step TAP AMG with block vectors
    ml = new ParSmoothedAggregationSolver(strong_threshold);
    ml->tap_amg = 0;
    ml->setup(A, nrhs);

    ml_single = new ParSmoothedAggregationSolver(strong_threshold);
    ml_single->tap_amg = 0;
    ml_single->setup(A);

    // Delete 3-step tap comms and replace with 2-step
    for (int i = 0; i < ml->num_levels; i++)
    {
        if (ml->levels[i]->A->tap_comm)
        {
            delete ml->levels[i]->A->tap_comm;
            ml->levels[i]->A->tap_comm = new TAPComm(ml->levels[i]->A->partition,
                    ml->levels[i]->A->off_proc_column_map, ml->levels[i]->A->on_proc_column_map, false);
        }
        if (ml->levels[i]->P)
        {
            if (ml->levels[i]->P->tap_comm)
            {
                delete ml->levels[i]->P->tap_comm;
                ml->levels[i]->P->tap_comm = new TAPComm(ml->levels[i]->P->partition,
                        ml->levels[i]->P->off_proc_column_map, ml->levels[i]->P->on_proc_column_map, false);
            }
        }
    }
   
    for (int i = 0; i < ml_single->num_levels; i++)
    {
        if (ml_single->levels[i]->A->tap_comm)
        {
            delete ml_single->levels[i]->A->tap_comm;
            ml_single->levels[i]->A->tap_comm = new TAPComm(ml_single->levels[i]->A->partition,
                    ml_single->levels[i]->A->off_proc_column_map, ml_single->levels[i]->A->on_proc_column_map, false);
        }
        if (ml_single->levels[i]->P)
        {
            if (ml_single->levels[i]->P->tap_comm)
            {
                delete ml_single->levels[i]->P->tap_comm;
                ml_single->levels[i]->P->tap_comm = new TAPComm(ml_single->levels[i]->P->partition,
                        ml_single->levels[i]->P->off_proc_column_map, ml_single->levels[i]->P->on_proc_column_map, false);
            }
        }
    }
    
    x.set_const_value(1.0);
    x.scale(1.0, &(alphas[0]));

    A->mult(x, b);
    x.set_const_value(0.0);
    ml->cycle(x, b);

    x_single.set_const_value(1.0); 
    A->mult(x_single, b_single);
    x_single.set_const_value(0.0);
    ml_single->cycle(x_single, b_single);
    
    // Compare values in x_single and x
    for (int i = 0; i < x.local_n; i++)
    {
        ASSERT_NEAR(x.local->values[i], x_single.local->values[i], 1e-10);
    }
   
    x_single.set_const_value(2.0); 
    A->mult(x_single, b_single);
    x_single.set_const_value(0.0);
    ml_single->cycle(x_single, b_single);
    
    // Compare values in x_single and x
    for (int i = 0; i < x.local_n; i++)
    {
        ASSERT_NEAR(x.local->values[x.local_n + i], x_single.local->values[i], 1e-10);
    }
    
    x_single.set_const_value(3.0); 
    A->mult(x_single, b_single);
    x_single.set_const_value(0.0);
    ml_single->cycle(x_single, b_single);
    
    // Compare values in x_single and x
    for (int i = 0; i < x.local_n; i++)
    {
        ASSERT_NEAR(x.local->values[2*x.local_n + i], x_single.local->values[i], 1e-10);
    }

    delete ml;
    delete ml_single;
    
    // ******************************************
    // NEED TO CHECK WHERE TAP SOLVE IS BREAKING
    // ******************************************
    
    /*x.set_const_value(0.0);
    x_single.set_const_value(0.0);

    iter_block = ml->solve(x, b);
    iter_single = ml_single->solve(x_single, b_single);*/

    // Check residuals
    /*aligned_vector<double>& mrhs_res = ml->get_residuals();
    aligned_vector<double>& single_res = ml_single->get_residuals();
    for (int i = 0; i < iter_single; i++)
    {
        for (int v = 0; v < nrhs; v++)
        {
            ASSERT_NEAR(mrhs_res[i*nrhs + v], single_res[i], 1e-10);
        }
    }*/
   
    delete A;
    
    setenv("PPN", "16", 1);

} // end of TEST(ParBlockSmoothAggTest, TestsInMultilevel) //
