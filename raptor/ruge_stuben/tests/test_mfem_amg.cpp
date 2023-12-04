// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include <iostream>
#include <fstream>

#include "gtest/gtest.h"
#include "mpi.h"

#include "raptor/raptor.hpp"
#include "raptor/tests/hypre_compare.hpp"

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

TEST(TestHypreAgg, TestsInRuge_Stuben)
{ 
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    FILE* f;
    int cf;

    double strong_threshold = 0.25;
    int hyp_coarsen_type = 8; // PMIS 
    int hyp_interp_type = 6; // Extended
    int p_max_elmts = 0;
    int agg_num_levels = 0;
    ParCSRMatrix* A;
    ParVector x, b;
    ParMultilevel* ml;
    HYPRE_IJMatrix Aij;
    hypre_ParCSRMatrix* A_hyp;
    int n;
    std::vector<int> grid;
    double* stencil;

    hypre_ParVector* x_hyp;
    hypre_ParVector* b_hyp;
    HYPRE_IJVector x_h_ij;
    HYPRE_IJVector b_h_ij;
    HYPRE_Solver solver_data;
    hypre_ParCSRMatrix** A_array;
    hypre_ParCSRMatrix** P_array;

#ifdef USING_MFEM
    /************************************
     **** Test Anisotropic Diffusion 
     ***********************************/
    std::string mesh_file = std::string(MFEM_MESH_DIR) + "/star-surf.mesh";
    int order = 3;
    int seq_refines = 4;
    int par_refines = 0;
    A = mfem_grad_div(x, b, mesh_file.c_str(), order, seq_refines, par_refines);

    ml = new ParRugeStubenSolver(strong_threshold, PMIS, Extended, Classical, SOR);
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

    double* weights;
    form_hypre_weights(&weights, A->local_num_rows);
    for (int level = 0; level < ml->num_levels - 1; level++) 
    {
        ParCSRMatrix* Al = ml->levels[level]->A;

        HYPRE_IJMatrix Al_ij;
        hypre_ParCSRMatrix* A_h;
        Al_ij = convert(Al);
        HYPRE_IJMatrixGetObject(Al_ij, (void**) &A_h);
//        compare(Al, A_h);

        hypre_ParCSRMatrix* S_h;
        hypre_BoomerAMGCreateS(A_h, strong_threshold, 1.0, 1, NULL, &S_h);
        ParCSRMatrix* S = Al->strength(Classical, strong_threshold);
//        compareS(S, S_h);

        int* CF_marker;
        std::vector<int> states;
        std::vector<int> off_proc_states;
        hypre_BoomerAMGCoarsenPMIS(S_h, A_h, 0, 0, &CF_marker);
        split_pmis(S, states, off_proc_states, false, weights);
        compare_states(Al->local_num_rows, states, CF_marker);

        int* coarse_dof_func = NULL;
        int* coarse_pnts_global = NULL;
        hypre_ParCSRMatrix* P_h;
        int local_num_vars = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_h));
        hypre_BoomerAMGCoarseParms(MPI_COMM_WORLD, local_num_vars, 1,
                NULL, CF_marker, &coarse_dof_func, &coarse_pnts_global);
        hypre_BoomerAMGBuildExtPIInterp(A_h, CF_marker, S_h, coarse_pnts_global,
                1, NULL, 0, 0.3, 0, NULL, &P_h);
        ParCSRMatrix* P = extended_interpolation(Al, S, states, off_proc_states, 0.3);
        compare(P, P_h);

        hypre_ParCSRMatrix* A_h_c;
        hypre_BoomerAMGBuildCoarseOperatorKT(P_h, A_h, P_h, false, &A_h_c);
        ParCSRMatrix* AP = Al->mult(P);
        ParCSCMatrix* P_csc = P->to_ParCSC();
        ParCSRMatrix* Ac = AP->mult_T(P_csc);

        hypre_ParCSRMatrixDestroy(A_h_c);
        hypre_ParCSRMatrixDestroy(P_h);
        hypre_ParCSRMatrixDestroy(S_h);
        HYPRE_IJMatrixDestroy(Al_ij);
        delete[] CF_marker;

        delete Ac;
        delete P_csc;
        delete AP;
        delete P;
        delete S;

        compare(ml->levels[level]->P, P_array[level]);
        compare(ml->levels[level+1]->A, A_array[level+1]);
    }
    delete[] weights;

    hypre_BoomerAMGDestroy(solver_data);
    HYPRE_IJMatrixDestroy(Aij);
    HYPRE_IJVectorDestroy(x_h_ij);
    HYPRE_IJVectorDestroy(b_h_ij);
    delete ml;
    delete A;
#endif


} // end of TEST(TestHypreAgg, TestsInRuge_Stuben) //








