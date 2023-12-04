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

TEST(TestParSplitting, TestsInRuge_Stuben)
{ 
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    FILE* f;
    std::vector<int> states;
    std::vector<int> off_proc_states;
    int cf;

    ParCSRMatrix* A;
    ParCSRMatrix* S;
    ParCSRMatrix* S_rap;
    HYPRE_IJMatrix Aij;
    hypre_ParCSRMatrix* A_hyp;
    hypre_ParCSRMatrix* S_hyp;

    const char* A0_fn = "../../../../test_data/rss_A0.pm";
    const char* A1_fn = "../../../../test_data/rss_A1.pm";


    // TEST LEVEL 0
    A = readParMatrix(A0_fn);
    Aij = convert(A);
    HYPRE_IJMatrixGetObject(Aij, (void**) &A_hyp);
    compare(A, A_hyp);
    
    S_rap = A->strength(Classical, 0.25);
    hypre_BoomerAMGCreateS(A_hyp, 0.25, 1.0, 1, NULL, &S_hyp);
    compareS(S_rap, S_hyp);
    
    HYPRE_IJMatrixDestroy(Aij);
    hypre_ParCSRMatrixDestroy(S_hyp);

    delete S_rap;
    delete A;



    // TEST LEVEL 1
    A = readParMatrix(A1_fn);
    Aij = convert(A);
    HYPRE_IJMatrixGetObject(Aij, (void**) &A_hyp);
    compare(A, A_hyp);
    
    S_rap = A->strength(Classical, 0.25);
    hypre_BoomerAMGCreateS(A_hyp, 0.25, 1.0, 1, NULL, &S_hyp); 
    compareS(S_rap, S_hyp);
    
    HYPRE_IJMatrixDestroy(Aij);
    hypre_ParCSRMatrixDestroy(S_hyp);

    delete S_rap;
    delete A;


} // end of TEST(TestParSplitting, TestsInRuge_Stuben) //


