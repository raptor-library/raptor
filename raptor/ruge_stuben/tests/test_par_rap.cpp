// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "tests/par_compare.hpp"

using namespace raptor;
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp = RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //

TEST(TestParRAP, TestsInRuge_Stuben)
{ 
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    ParCSRMatrix* A;
    ParCSRMatrix* P;
    ParCSRMatrix* Ac;
    ParCSRMatrix* AP;
    ParCSCMatrix* P_csc;
    ParCSRMatrix* Ac_rap;

    const char* A0_fn = "../../../../test_data/rss_A0.pm";
    const char* A1_fn = "../../../../test_data/rss_A1.pm";
    const char* A2_fn = "../../../../test_data/rss_A2.pm";
    const char* P0_fn = "../../../../test_data/rss_P0.pm";
    const char* P1_fn = "../../../../test_data/rss_P1.pm";

    // TEST LEVEL 0
    A = readParMatrix(A0_fn);
    P = readParMatrix(P0_fn);
    AP = A->mult(P);
    P_csc = P->to_ParCSC();
    Ac = AP->mult_T(P_csc);
    Ac_rap = readParMatrix(A1_fn);
    compare(Ac, Ac_rap);
    delete Ac_rap;
    delete Ac;
    delete P_csc;
    delete AP;
    delete P;
    delete A;

    // TEST LEVEL 1
    A = readParMatrix(A1_fn);
    P = readParMatrix(P1_fn);
    AP = A->mult(P);
    P_csc = P->to_ParCSC();
    Ac = AP->mult_T(P_csc);
    Ac_rap = readParMatrix(A2_fn);
    compare(Ac, Ac_rap);
    delete Ac_rap;
    delete Ac;
    delete P_csc;
    delete AP;
    delete P;
    delete A;
} // end of TEST(TestParRAP, TestsInRuge_Stuben) //
