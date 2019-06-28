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

TEST(TestTAPRAP, TestsInRuge_Stuben)
{ 
    setenv("PPN", "4", 1);

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
    P_csc = P->to_ParCSC();

    A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map, true);
    AP = A->tap_mult(P);
    Ac = AP->tap_mult_T(P_csc);
    Ac_rap = readParMatrix(A1_fn);
    compare(Ac, Ac_rap);
    delete Ac_rap;
    delete Ac;
    delete AP;

    delete A->tap_comm;

    A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map, false);
    AP = A->tap_mult(P);
    Ac = AP->tap_mult_T(P_csc);
    Ac_rap = readParMatrix(A1_fn);
    compare(Ac, Ac_rap);
    delete Ac_rap;
    delete Ac;
    delete AP;

    delete P_csc;
    delete P;
    delete A;

    // TEST LEVEL 1
    A = readParMatrix(A1_fn);
    P = readParMatrix(P1_fn);
    P_csc = P->to_ParCSC();

    A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map, true);
    AP = A->tap_mult(P);
    Ac = AP->tap_mult_T(P_csc);
    Ac_rap = readParMatrix(A2_fn);
    compare(Ac, Ac_rap);
    delete Ac_rap;
    delete Ac;
    delete AP;

    delete A->tap_comm;

    A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map, false);
    AP = A->tap_mult(P);
    Ac = AP->tap_mult_T(P_csc);
    Ac_rap = readParMatrix(A2_fn);
    compare(Ac, Ac_rap);
    delete Ac_rap;
    delete Ac;
    delete AP;

    delete P_csc;
    delete P;
    delete A;

    setenv("PPN", "16", 1);
} // end of TEST(TestParRAP, TestsInRuge_Stuben) //
