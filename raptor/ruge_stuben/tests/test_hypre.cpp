// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause


#include "gtest/gtest.h"
#include "raptor.hpp"
#include "tests/hypre_compare.hpp"
#include <iostream>
#include <fstream>

using namespace raptor;

void form_hypre_weights(double** weight_ptr, int n_rows)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    hypre_SeedRand(2747 + rank);
    double* weights;
    if (n_rows)
    {
        weights = new double[n_rows];
        for (int i = 0; i < n_rows; i++)
        {
            weights[i] = hypre_Rand();
        }
    }

    *weight_ptr = weights;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp = RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //

TEST(TestHypre, TestsInRuge_Stuben)
{ 
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    FILE* f;
    int cf;

    double strong_threshold = 0.25;
    int hyp_coarsen_type = 0; // CLJP 
    int hyp_interp_type = 0; // ModClassical 
    int p_max_elmts = 0;
    int agg_num_levels = 0;
    ParCSRMatrix* A;
    ParVector x, b;
    ParMultilevel* ml;
    HYPRE_IJMatrix Aij;
    hypre_ParCSRMatrix* A_hyp;
    int n;
    aligned_vector<int> grid;
    double* stencil;


    /************************************
     **** Test Laplacian 
     ***********************************/
    n = 25;
    grid.resize(3);
    std::fill(grid.begin(), grid.end(), n);
    stencil = laplace_stencil_27pt();
    A = par_stencil_grid(stencil, grid.data(), 3);
    x = ParVector(A->global_num_rows, A->local_num_rows);
    b = ParVector(A->global_num_rows, A->local_num_rows);
    delete[] stencil;

    ml = new ParRugeStubenSolver(strong_threshold, CLJP, ModClassical, Classical, SOR);
    form_hypre_weights(&ml->weights, A->local_num_rows);
    ml->setup(A);

    Aij = convert(A);
    HYPRE_IJMatrixGetObject(Aij, (void**) &A_hyp);
    compare(A, A_hyp);

    // Convert vectors... needed for Hypre Setup
    hypre_ParVector* x_hyp;
    hypre_ParVector* b_hyp;
    HYPRE_IJVector x_h_ij = convert(x);
    HYPRE_IJVector b_h_ij = convert(b);
    HYPRE_IJVectorGetObject(x_h_ij, (void **) &x_hyp);
    HYPRE_IJVectorGetObject(b_h_ij, (void **) &b_hyp);

    // Setup Hypre Hierarchy
    HYPRE_Solver solver_data = hypre_create_hierarchy(A_hyp, x_hyp, b_hyp, 
            hyp_coarsen_type, hyp_interp_type, p_max_elmts, agg_num_levels, 
            strong_threshold);

    hypre_ParCSRMatrix** A_array = hypre_ParAMGDataAArray((hypre_ParAMGData*) solver_data);
    hypre_ParCSRMatrix** P_array = hypre_ParAMGDataPArray((hypre_ParAMGData*) solver_data);

    for (int level = 0; level < ml->num_levels - 1; level++) 
    {
        compare(ml->levels[level]->P, P_array[level]);
        compare(ml->levels[level+1]->A, A_array[level+1]);
    }

    hypre_BoomerAMGDestroy(solver_data);    
    HYPRE_IJMatrixDestroy(Aij);
    delete ml;
    delete A;


    /************************************
     **** Test Anisotropic Diffusion 
     ***********************************/
    n = 100;
    grid.resize(2);
    std::fill(grid.begin(), grid.end(), n);
    stencil = diffusion_stencil_2d(0.001, M_PI/4.0);
    A = par_stencil_grid(stencil, grid.data(), 2);
    x = ParVector(A->global_num_rows, A->local_num_rows);
    b = ParVector(A->global_num_rows, A->local_num_rows);
    delete[] stencil;

    ml = new ParRugeStubenSolver(strong_threshold, CLJP, ModClassical, Classical, SOR);
    form_hypre_weights(&ml->weights, A->local_num_rows);
    ml->setup(A);

    Aij = convert(A);
    HYPRE_IJMatrixGetObject(Aij, (void**) &A_hyp);
    compare(A, A_hyp);

    // Convert vectors... needed for Hypre Setup
    x_h_ij = convert(x);
    b_h_ij = convert(b);
    HYPRE_IJVectorGetObject(x_h_ij, (void **) &x_hyp);
    HYPRE_IJVectorGetObject(b_h_ij, (void **) &b_hyp);

    // Setup Hypre Hierarchy
    solver_data = hypre_create_hierarchy(A_hyp, x_hyp, b_hyp, 
            hyp_coarsen_type, hyp_interp_type, p_max_elmts, agg_num_levels, 
            strong_threshold);

    A_array = hypre_ParAMGDataAArray((hypre_ParAMGData*) solver_data);
    P_array = hypre_ParAMGDataPArray((hypre_ParAMGData*) solver_data);

    for (int level = 0; level < ml->num_levels - 1; level++) 
    {
        compare(ml->levels[level]->P, P_array[level]);
        compare(ml->levels[level+1]->A, A_array[level+1]);
    }

    hypre_BoomerAMGDestroy(solver_data);    
    HYPRE_IJMatrixDestroy(Aij);
    delete ml;
    delete A;


} // end of TEST(TestHypre, TestsInRuge_Stuben) //







