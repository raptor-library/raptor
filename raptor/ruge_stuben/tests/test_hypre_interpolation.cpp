// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include <iostream>
#include <fstream>

#include "gtest/gtest.h"

#include "raptor/raptor.hpp"
#include "raptor/tests/hypre_compare.hpp"

using namespace raptor;

void form_hypre_weights(std::vector<double> & weights, int n_rows)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    hypre_SeedRand(2747 + rank);
    weights.resize(n_rows);
    for (int i = 0; i < n_rows; i++)
    {
	    weights[i] = hypre_Rand();
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp = RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //

TEST(TestHypreInterpolation, TestsInRuge_Stuben)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::vector<int> states;
    std::vector<int> off_proc_states;
    std::vector<double> weights;

    ParCSRMatrix* A;
    ParCSRMatrix* S;
    ParCSRMatrix* P;
    HYPRE_IJMatrix Aij;
    hypre_ParCSRMatrix* A_hyp;
    hypre_ParCSRMatrix* S_hyp;
    hypre_ParCSRMatrix* P_hyp;
    hypre_IntArray* states_hypre = NULL;
    hypre_IntArray* coarse_dof_func = NULL;
    std::vector<HYPRE_BigInt> coarse_pnts_gbl;
    coarse_pnts_gbl.resize(num_procs + 1);

    const char* A0_fn = "../../../../test_data/rss_A0.pm";
    const char* A1_fn = "../../../../test_data/rss_A1.pm";


    // TEST LEVEL 0
    A = readParMatrix(A0_fn);
    form_hypre_weights(weights, A->local_num_rows);
    Aij = convert(A);
    HYPRE_IJMatrixGetObject(Aij, (void**) &A_hyp);
    S = A->strength(Classical, 0.25);
    hypre_BoomerAMGCreateS(A_hyp, 0.25, 1.0, 1, NULL, &S_hyp);
    compareS(S, S_hyp);

    split_pmis(S, states, off_proc_states, false, weights.data());
    hypre_BoomerAMGCoarsenPMIS(S_hyp, A_hyp, 0, 0, &states_hypre);
    compare_states(A->local_num_rows, states, states_hypre);

    // Modified Classical Interpolation
    hypre_BoomerAMGCoarseParms(MPI_COMM_WORLD, A->local_num_rows, 1, NULL, states_hypre,
                               &coarse_dof_func, coarse_pnts_gbl.data());
    P = mod_classical_interpolation(A, S, states, off_proc_states, false);
    hypre_BoomerAMGBuildInterp(A_hyp, hypre_IntArrayData(states_hypre), S_hyp, coarse_pnts_gbl.data(), 1, NULL, 0, 0.0, 0, &P_hyp);
    compare(P, P_hyp);
    hypre_ParCSRMatrixDestroy(P_hyp);
    delete P;

    // Extended+i Interpolation
    hypre_BoomerAMGCoarseParms(MPI_COMM_WORLD, A->local_num_rows, 1, NULL, states_hypre,
                               &coarse_dof_func, coarse_pnts_gbl.data());
    P = extended_interpolation(A, S, states, off_proc_states, 0.3, false);
    hypre_BoomerAMGBuildExtPIInterp(A_hyp, hypre_IntArrayData(states_hypre), S_hyp, coarse_pnts_gbl.data(), 1, NULL, 0, 0.3, 0, &P_hyp);
    compare(P, P_hyp);
    hypre_ParCSRMatrixDestroy(P_hyp);
    delete P;

    hypre_IntArrayDestroy(states_hypre);
    states_hypre = NULL;
    HYPRE_IJMatrixDestroy(Aij);
    hypre_ParCSRMatrixDestroy(S_hyp);
    delete S;
    delete A;


    // TEST LEVEL 1
    A = readParMatrix(A1_fn);
    form_hypre_weights(weights, A->local_num_rows);
    Aij = convert(A);
    HYPRE_IJMatrixGetObject(Aij, (void**) &A_hyp);
    S = A->strength(Classical, 0.25);
    hypre_BoomerAMGCreateS(A_hyp, 0.25, 1.0, 1, NULL, &S_hyp);
    split_pmis(S, states, off_proc_states, false, weights.data());
    hypre_BoomerAMGCoarsenPMIS(S_hyp, A_hyp, 0, 0, &states_hypre);

    // Modified Classical Interpolation
    hypre_BoomerAMGCoarseParms(MPI_COMM_WORLD, A->local_num_rows, 1, NULL, states_hypre,
                               &coarse_dof_func, coarse_pnts_gbl.data());
    P = mod_classical_interpolation(A, S, states, off_proc_states, false);
    hypre_BoomerAMGBuildInterp(A_hyp, hypre_IntArrayData(states_hypre), S_hyp, coarse_pnts_gbl.data(), 1, NULL, 0, 0.0, 0, &P_hyp);
    compare(P, P_hyp);
    hypre_ParCSRMatrixDestroy(P_hyp);
    delete P;

    // Extended+i Interpolation
    hypre_BoomerAMGCoarseParms(MPI_COMM_WORLD, A->local_num_rows, 1, NULL, states_hypre,
                               &coarse_dof_func, coarse_pnts_gbl.data());
    P = extended_interpolation(A, S, states, off_proc_states, 0.3, false);
    hypre_BoomerAMGBuildExtPIInterp(A_hyp, hypre_IntArrayData(states_hypre), S_hyp, coarse_pnts_gbl.data(), 1, NULL, 0, 0.3, 0, &P_hyp);
    compare(P, P_hyp);
    hypre_ParCSRMatrixDestroy(P_hyp);
    delete P;

    hypre_IntArrayDestroy(states_hypre);
    HYPRE_IJMatrixDestroy(Aij);
    hypre_ParCSRMatrixDestroy(S_hyp);
    delete S;
    delete A;
} // end of TEST(TestParSplitting, TestsInRuge_Stuben) //
