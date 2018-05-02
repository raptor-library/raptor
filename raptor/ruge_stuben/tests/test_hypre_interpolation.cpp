// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause


#include "gtest/gtest.h"
#include "mpi.h"
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "gallery/external/hypre_wrapper.hpp"
#include "ruge_stuben/par_cf_splitting.hpp"
#include "ruge_stuben/par_interpolation.hpp"
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

TEST(TestParSplitting, TestsInRuge_Stuben)
{ 
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    FILE* f;
    aligned_vector<int> states;
    aligned_vector<int> off_proc_states;
    double* weights;
    int cf;

    ParCSRMatrix* A;
    ParCSRMatrix* S;
    ParCSRMatrix* P;
    HYPRE_IJMatrix Aij;
    hypre_ParCSRMatrix* A_hyp;
    hypre_ParCSRMatrix* S_hyp;
    hypre_ParCSRMatrix* P_hyp;
    int* states_hypre;
    int* coarse_dof_func;
    int* coarse_pnts_gbl;

    const char* A0_fn = "../../../../test_data/rss_A0.pm";
    const char* A1_fn = "../../../../test_data/rss_A1.pm";


    // TEST LEVEL 0
    A = readParMatrix(A0_fn);
    form_hypre_weights(&weights, A->local_num_rows);
    Aij = convert(A);
    HYPRE_IJMatrixGetObject(Aij, (void**) &A_hyp);
    S = A->strength(0.25);
    hypre_BoomerAMGCreateS(A_hyp, 0.25, 1.0, 1, NULL, &S_hyp);
    split_pmis(S, states, off_proc_states, false, weights);
    hypre_BoomerAMGCoarsenPMIS(S_hyp, A_hyp, 0, 0, &states_hypre);
    
    // Modified Classical Interpolation
    hypre_BoomerAMGCoarseParms(MPI_COMM_WORLD, A->local_num_rows, 1, NULL, states_hypre,
            &coarse_dof_func, &coarse_pnts_gbl);
    P = mod_classical_interpolation(A, S, states, off_proc_states, false);
    hypre_BoomerAMGBuildInterp(A_hyp, states_hypre, S_hyp, coarse_pnts_gbl, 1, NULL, 0, 0.0, 0.0, NULL, &P_hyp);
    compare(P, P_hyp);
    hypre_ParCSRMatrixDestroy(P_hyp);
    delete P;
    
    // Extended+i Interpolation
    hypre_BoomerAMGCoarseParms(MPI_COMM_WORLD, A->local_num_rows, 1, NULL, states_hypre,
            &coarse_dof_func, &coarse_pnts_gbl);
    P = extended_interpolation(A, S, states, off_proc_states, false);
    hypre_BoomerAMGBuildExtPIInterp(A_hyp, states_hypre, S_hyp, coarse_pnts_gbl, 1, NULL, 0, 0.0, 0.0, NULL, &P_hyp);
    compare(P, P_hyp);
    hypre_ParCSRMatrixDestroy(P_hyp);
    delete P;

    hypre_TFree(states_hypre);
    HYPRE_IJMatrixDestroy(Aij);
    hypre_ParCSRMatrixDestroy(S_hyp);
    delete[] weights;
    delete S;
    delete A;


    // TEST LEVEL 1
    A = readParMatrix(A1_fn);
    form_hypre_weights(&weights, A->local_num_rows);
    Aij = convert(A);
    HYPRE_IJMatrixGetObject(Aij, (void**) &A_hyp);
    S = A->strength(0.25);
    hypre_BoomerAMGCreateS(A_hyp, 0.25, 1.0, 1, NULL, &S_hyp);
    split_pmis(S, states, off_proc_states, false, weights);
    hypre_BoomerAMGCoarsenPMIS(S_hyp, A_hyp, 0, 0, &states_hypre);
    
    // Modified Classical Interpolation
    hypre_BoomerAMGCoarseParms(MPI_COMM_WORLD, A->local_num_rows, 1, NULL, states_hypre,
            &coarse_dof_func, &coarse_pnts_gbl);
    P = mod_classical_interpolation(A, S, states, off_proc_states, false);
    hypre_BoomerAMGBuildInterp(A_hyp, states_hypre, S_hyp, coarse_pnts_gbl, 1, NULL, 0, 0.0, 0.0, NULL, &P_hyp);
    compare(P, P_hyp);
    hypre_ParCSRMatrixDestroy(P_hyp);
    delete P;
    
    // Extended+i Interpolation
    hypre_BoomerAMGCoarseParms(MPI_COMM_WORLD, A->local_num_rows, 1, NULL, states_hypre,
            &coarse_dof_func, &coarse_pnts_gbl);
    P = extended_interpolation(A, S, states, off_proc_states, false);
    hypre_BoomerAMGBuildExtPIInterp(A_hyp, states_hypre, S_hyp, coarse_pnts_gbl, 1, NULL, 0, 0.0, 0.0, NULL, &P_hyp);
    compare(P, P_hyp);
    hypre_ParCSRMatrixDestroy(P_hyp);
    delete P;

    hypre_TFree(states_hypre);
    HYPRE_IJMatrixDestroy(Aij);
    hypre_ParCSRMatrixDestroy(S_hyp);
    delete[] weights;
    delete S;
    delete A;



} // end of TEST(TestParSplitting, TestsInRuge_Stuben) //




