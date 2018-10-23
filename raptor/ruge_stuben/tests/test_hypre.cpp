// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause


#include "gtest/gtest.h"
#include "mpi.h"
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/par_stencil.hpp"
#include "gallery/laplacian27pt.hpp"
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

TEST(TestHypreAgg, TestsInRuge_Stuben)
{ 
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    FILE* f;
    double* weights;
    int cf;

    int n = 25;
    aligned_vector<int> grid(3, n);
    double* stencil = laplace_stencil_27pt();

    std::vector<ParCSRMatrix*> A_array;
    std::vector<ParCSRMatrix*> P_array;

    ParCSRMatrix* A = par_stencil_grid(stencil, grid.data(), 3);
    A_array.push_back(A);
    delete[] stencil;
    form_hypre_weights(&weights, A->local_num_rows);

    HYPRE_IJMatrix Aij;
    hypre_ParCSRMatrix* A_hyp;
    Aij = convert(A);
    HYPRE_IJMatrixGetObject(Aij, (void**) &A_hyp);
    compare(A, A_hyp);

    hypre_ParCSRMatrix* S_hyp;
    hypre_ParCSRMatrix* P_hyp;
    hypre_ParCSRMatrix* Ac_hyp;
    int* states_hypre;
    int* coarse_dof_func;
    int* coarse_pnts_gbl;

    aligned_vector<int> states;
    aligned_vector<int> off_proc_states;

    int nrows = A_array[0]->global_num_rows;
    int level = 0;
    while (nrows > 50)
    {
        ParCSRMatrix* Al = A_array[level];

        // Create Strength of Connection Matrix
        ParCSRMatrix* Sl = Al->strength(Classical, 0.25);
        hypre_BoomerAMGCreateS(A_hyp, 0.25, 1.0, 1, NULL, &S_hyp);
        compareS(Sl, S_hyp);

        // C/F Splitting (PMIS)
        split_cljp(Sl, states, off_proc_states, false, weights);
        hypre_BoomerAMGCoarsen(S_hyp, A_hyp, 0, 0, &states_hypre);
        compare_states(Al->local_num_rows, states, states_hypre);

        // Extended Interpolation
        ParCSRMatrix* Pl = mod_classical_interpolation(Al, Sl, states, off_proc_states, false);
        P_array.push_back(Pl);
        hypre_BoomerAMGCoarseParms(MPI_COMM_WORLD, Al->local_num_rows, 1, NULL, states_hypre,
                &coarse_dof_func, &coarse_pnts_gbl);
        hypre_BoomerAMGBuildInterp(A_hyp, states_hypre, S_hyp, coarse_pnts_gbl, 1, NULL, 
                0, 0.0, 0.0, NULL, &P_hyp);
        compare(Pl, P_hyp);

        // SpGEMM (Form Ac)
        ParCSRMatrix* APl = Al->mult(Pl);
	    ParCSCMatrix* Pcsc = Pl->to_ParCSC();
	    APl->comm = new ParComm(APl->partition, APl->off_proc_column_map, APl->on_proc_column_map);
        ParCSRMatrix* Ac = APl->mult_T(Pcsc);
        Ac->comm = new ParComm(Ac->partition, Ac->off_proc_column_map, Ac->on_proc_column_map);
        A_array.push_back(Ac);
        hypre_BoomerAMGBuildCoarseOperator(P_hyp, A_hyp, P_hyp, &Ac_hyp);
        compare(Ac, Ac_hyp);
                
        if (level > 0)
            hypre_ParCSRMatrixDestroy(A_hyp);
        A_hyp = Ac_hyp;

        nrows = Ac->global_num_rows;
        level++;

        hypre_TFree(states_hypre, HYPRE_MEMORY_HOST);
        hypre_ParCSRMatrixDestroy(P_hyp);
        hypre_ParCSRMatrixDestroy(S_hyp);

        delete APl;
        delete Pcsc;
        delete Sl;
    }
    hypre_ParCSRMatrixDestroy(Ac_hyp);

    for (std::vector<ParCSRMatrix*>::iterator it = A_array.begin(); it != A_array.end(); ++it)
        delete *it;

    for (std::vector<ParCSRMatrix*>::iterator it = P_array.begin(); it != P_array.end(); ++it)
        delete *it;

    delete[] weights;

    HYPRE_IJMatrixDestroy(Aij);


} // end of TEST(TestParSplitting, TestsInRuge_Stuben) //







