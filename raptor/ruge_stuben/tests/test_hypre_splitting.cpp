// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause


#include "gtest/gtest.h"
#include "mpi.h"
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "gallery/external/hypre_wrapper.hpp"
#include "ruge_stuben/par_cf_splitting.hpp"
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
    ParCSRMatrix* S_rap;
    HYPRE_IJMatrix Aij;
    hypre_ParCSRMatrix* A_hyp;
    hypre_ParCSRMatrix* S_hyp;
    int* states_hypre;

    const char* A0_fn = "../../../../test_data/rss_A0.pm";
    const char* A1_fn = "../../../../test_data/rss_A1.pm";


    // TEST LEVEL 0
    A = readParMatrix(A0_fn);
    form_hypre_weights(&weights, A->local_num_rows);
    Aij = convert(A);
    HYPRE_IJMatrixGetObject(Aij, (void**) &A_hyp);
    compare(A, A_hyp);
    S_rap = A->strength(Classical, 0.25);
    hypre_BoomerAMGCreateS(A_hyp, 0.25, 1.0, 1, NULL, &S_hyp);
    
    // Ruge
    // p


    // CLJP
    split_cljp(S_rap, states, off_proc_states, false, weights);
    hypre_BoomerAMGCoarsen(S_hyp, A_hyp, 0, 0, &states_hypre);
    for (int i = 0; i < states.size(); i++)
    {
        if (states[i] == 1)
        {
            ASSERT_EQ(states[i], states_hypre[i]);
        }
        else
        {
            ASSERT_EQ(states[i], 0);
            ASSERT_EQ(states_hypre[i], -1);
        }
    }
    hypre_TFree(states_hypre);

    // Falgout
    // TODO --Ruge stuben doesnt match hypre
/*    split_falgout(S_rap, states, off_proc_states, weights);
    hypre_BoomerAMGCoarsenFalgout(S_hyp, A_hyp, 0, 0, &states_hypre);
    for (int i = 0; i < states.size(); i++)
    {
        if (states[i] == 1)
        {
            ASSERT_EQ(states[i], states_hypre[i]);
        }
        else
        {
            ASSERT_EQ(states[i], 0);
            ASSERT_EQ(states_hypre[i], -1);
        }
    }
    hypre_TFree(states_hypre);
    */

    // PMIS
    split_pmis(S_rap, states, off_proc_states, false, weights);
    hypre_BoomerAMGCoarsenPMIS(S_hyp, A_hyp, 0, 0, &states_hypre);
    for (int i = 0; i < states.size(); i++)
    {
        if (states[i] == 1)
        {
            ASSERT_EQ(states[i], states_hypre[i]);
        }
        else
        {
            ASSERT_EQ(states[i], 0);
            ASSERT_EQ(states_hypre[i], -1);
        }
    }
    hypre_TFree(states_hypre);

    // HMIS -- TODO
    // TODO --Ruge stuben doesnt match hypre
/*    split_hmis(S_rap, states, off_proc_states, weights);
    hypre_BoomerAMGCoarsenHMIS(S_hyp, A_hyp, 0, 0, &states_hypre);
    for (int i = 0; i < states.size(); i++)
    {
        if (states[i] == 1)
        {
            ASSERT_EQ(states[i], states_hypre[i]);
        }
        else
        {
            ASSERT_EQ(states[i], 0);
            ASSERT_EQ(states_hypre[i], -1);
        }
    }
    hypre_TFree(states_hypre);
*/


    HYPRE_IJMatrixDestroy(Aij);
    hypre_ParCSRMatrixDestroy(S_hyp);
    delete[] weights;
    delete S_rap;
    delete A;



    // TEST LEVEL 1
    A = readParMatrix(A1_fn);
    form_hypre_weights(&weights, A->local_num_rows);
    Aij = convert(A);
    HYPRE_IJMatrixGetObject(Aij, (void**) &A_hyp);
    compare(A, A_hyp);
    S_rap = A->strength(Classical, 0.25);
    hypre_BoomerAMGCreateS(A_hyp, 0.25, 1.0, 1, NULL, &S_hyp);
    
    // Ruge
    // TODO - implementation doesnt match hypre

    // CLJP
    split_cljp(S_rap, states, off_proc_states, false, weights);
    hypre_BoomerAMGCoarsen(S_hyp, A_hyp, 0, 0, &states_hypre);

    for (int i = 0; i < states.size(); i++)
    {
        if (states[i] == 1)
        {
            ASSERT_EQ(states[i], states_hypre[i]);
        }
        else
        {
            ASSERT_EQ(states[i], 0);
            ASSERT_EQ(states_hypre[i], -1);
        }
    }
    hypre_TFree(states_hypre);

    // Falgout
    // TODO - RS implementation doesnt match hypre


    // PMIS
    split_pmis(S_rap, states, off_proc_states, false, weights);
    hypre_BoomerAMGCoarsenPMIS(S_hyp, A_hyp, 0, 0, &states_hypre);
    for (int i = 0; i < states.size(); i++)
    {
        if (states[i] == 1)
        {
            ASSERT_EQ(states[i], states_hypre[i]);
        }
        else
        {
            ASSERT_EQ(states[i], 0);
            ASSERT_EQ(states_hypre[i], -1);
        }
    }
    hypre_TFree(states_hypre);

    // HMIS
    // TODO - RS implementation doesnt match hypre

 

    HYPRE_IJMatrixDestroy(Aij);
    hypre_ParCSRMatrixDestroy(S_hyp);
    delete[] weights;
    delete S_rap;
    delete A;


} // end of TEST(TestParSplitting, TestsInRuge_Stuben) //



