// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include <iostream>
#include <fstream>

#include "gtest/gtest.h"

#include "raptor/raptor.hpp"
#include "raptor/tests/hypre_compare.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp = RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //

int* create_unknown_variables(int local_n, int first_n, int num_var)
{
    if (local_n == 0 || num_var <= 1) return NULL;

    int* variables = new int[local_n];
    int remain = first_n - ((first_n / num_var) * num_var);
    int idx = num_var - remain;
    if (remain == 0) idx = 0;
    int k = num_var - 1;
    for (int j = idx - 1; j > -1; j--)
    {
        variables[j] = k--;
    }
    int tms = local_n / num_var;
    if (tms*num_var + idx > local_n) tms--;
    for (int j = 0; j < tms; j++)
    {
        for (k = 0; k < num_var; k++)
        {
            variables[idx++] = k;
        }
    }
    k = 0;
    while (idx < local_n)
    {
        variables[idx++] = k++;
    }

    return variables;
}

void form_hypre_weights(double** weight_ptr, int n_rows)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    hypre_SeedRand(2747 + rank);
    double* weights = nullptr;
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

TEST(TestParSplitting, TestsInRuge_Stuben)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    ParCSRMatrix* A;
    ParCSRMatrix* S;
    ParCSRMatrix* P;

    HYPRE_IJMatrix Aij;
    hypre_ParCSRMatrix* A_hyp;
    hypre_ParCSRMatrix* S_hyp;
    hypre_ParCSRMatrix* P_hyp;

    hypre_IntArray *states_hypre = NULL;
    hypre_IntArray *coarse_dof_func = NULL;
    std::vector<HYPRE_BigInt> coarse_pnts_gbl;
    coarse_pnts_gbl.resize(num_procs + 1);

    std::vector<int> states;
    std::vector<int> off_proc_states;
    double* weights;

    const char* A0_fn = "../../../../test_data/rss_A0.pm";
    const char* A1_fn = "../../../../test_data/rss_A1.pm";

    int num_variables = 2;
    int* var;

    // TEST LEVEL 0
    A = readParMatrix(A0_fn);
    form_hypre_weights(&weights, A->local_num_rows);
    Aij = convert(A);
    HYPRE_IJMatrixGetObject(Aij, (void**) &A_hyp);
    compare(A, A_hyp);

    // Create strength with unknown approach
    var = create_unknown_variables(A->local_num_rows,
            A->partition->first_local_row, num_variables);
    S = A->strength(Classical, 0.25, false, num_variables, var);
    hypre_BoomerAMGCreateS(A_hyp, 0.25, 1.0, num_variables, var, &S_hyp);
    compareS(S, S_hyp);

    // C/F Splitting as usual
    split_cljp(S, states, off_proc_states, false, weights);
    hypre_BoomerAMGCoarsen(S_hyp, A_hyp, 0, 0, &states_hypre);
    for (int i = 0; i < A->local_num_rows; i++)
    {
        if (states[i] == 1 || states[i] == -3)
        {
	        ASSERT_EQ(states[i], hypre_IntArrayData(states_hypre)[i]);
        }
        else
        {
            ASSERT_EQ(states[i], 0);
            ASSERT_EQ(hypre_IntArrayData(states_hypre)[i], -1);
        }
    }

    std::vector<int> coarse_variables;
    auto * hypre_var = hypre_IntArrayCreate(A->local_num_rows);
    hypre_IntArrayInitialize(hypre_var);
    for (int i = 0; i < A->local_num_rows; i++)
    {
        if (states[i] == 1)
            coarse_variables.push_back(var[i]);
        hypre_IntArrayData(hypre_var)[i] = var[i];
    }

    hypre_BoomerAMGCoarseParms(MPI_COMM_WORLD, A->local_num_rows, num_variables, hypre_var, states_hypre,
                               &coarse_dof_func, coarse_pnts_gbl.data());

    for (int i = 0; i < coarse_variables.size(); i++)
    {
	    ASSERT_EQ(coarse_variables[i], hypre_IntArrayData(coarse_dof_func)[i]);
    }

    P = mod_classical_interpolation(A, S, states, off_proc_states, false, num_variables, var);
    hypre_BoomerAMGBuildInterp(A_hyp, hypre_IntArrayData(states_hypre), S_hyp, coarse_pnts_gbl.data(), num_variables, var, 0, 0.0, 0.0, &P_hyp);
    compare(P, P_hyp);

    ParCSRMatrix* APtmp = A->mult(P);
    ParCSCMatrix* Pcsc = P->to_ParCSC();
    ParCSRMatrix* Ac = APtmp->mult_T(Pcsc);
    delete APtmp;
    delete Pcsc;
    hypre_ParCSRMatrix* A_H;
    hypre_BoomerAMGBuildCoarseOperator(P_hyp, A_hyp, P_hyp,  &A_H);
    compare(Ac, A_H);


    hypre_ParCSRMatrixDestroy(P_hyp);
    delete P;


    hypre_IntArrayDestroy(states_hypre);
    states_hypre = NULL;
    hypre_ParCSRMatrixDestroy(S_hyp);
    delete S;
    delete[] var;

    HYPRE_IJMatrixDestroy(Aij);
    delete A;



    // TEST LEVEL 1
    A = readParMatrix(A1_fn);
    Aij = convert(A);
    HYPRE_IJMatrixGetObject(Aij, (void**) &A_hyp);
    compare(A, A_hyp);

    num_variables = 2;
    var = create_unknown_variables(A->local_num_rows,
            A->partition->first_local_row, num_variables);
    S = A->strength(Classical, 0.25, false, num_variables, var);
    hypre_BoomerAMGCreateS(A_hyp, 0.25, 1.0, num_variables, var, &S_hyp);
    compareS(S, S_hyp);

    // C/F Splitting as usual
    split_pmis(S, states, off_proc_states, false, weights);
    hypre_BoomerAMGCoarsenPMIS(S_hyp, A_hyp, 0, 0, &states_hypre);
    for (int i = 0; i < A->local_num_rows; i++)
    {
        if (states[i] == 1 || states[i] == -3)
        {
	        ASSERT_EQ(states[i], hypre_IntArrayData(states_hypre)[i]);
        }
        else
        {
            ASSERT_EQ(states[i], 0);
            ASSERT_EQ(hypre_IntArrayData(states_hypre)[i], -1);
        }
        hypre_IntArrayData(hypre_var)[i] = var[i];
    }

    hypre_BoomerAMGCoarseParms(MPI_COMM_WORLD, A->local_num_rows, num_variables, hypre_var, states_hypre,
                               &coarse_dof_func, coarse_pnts_gbl.data());
    P = mod_classical_interpolation(A, S, states, off_proc_states, false, num_variables, var);
    hypre_BoomerAMGBuildInterp(A_hyp, hypre_IntArrayData(states_hypre), S_hyp, coarse_pnts_gbl.data(), num_variables, var, 0, 0.0, 0.0, &P_hyp);
    compare(P, P_hyp);
    hypre_ParCSRMatrixDestroy(P_hyp);
    delete P;

    hypre_IntArrayDestroy(states_hypre);
    hypre_ParCSRMatrixDestroy(S_hyp);
    delete S;
    delete[] var;
    hypre_IntArrayDestroy(hypre_var);

    HYPRE_IJMatrixDestroy(Aij);
    delete A;

    delete[] weights;


} // end of TEST(TestParSplitting, TestsInRuge_Stuben) //
