// EXPECT_EQ and ASSERT_EQ are macros
// EXPECT_EQ test execution and continues even if there is a failure
// ASSERT_EQ test execution and aborts if there is a failure
// The ASSERT_* variants abort the program execution if an assertion fails 
// while EXPECT_* variants continue with the run.


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

    // TEST LEVEL 0
    A = readParMatrix((char *)"../../../../test_data/rss_A0.mtx", MPI_COMM_WORLD, 1, 1);
    P = readParMatrix((char *)"../../../../test_data/rss_P0.mtx", MPI_COMM_WORLD, 1, 0);
    AP = A->mult(P);
    P_csc = new ParCSCMatrix(P);
    Ac = AP->mult_T(P_csc);
    Ac_rap = readParMatrix((char *)"../../../../test_data/rss_A1.mtx", MPI_COMM_WORLD, 1, 0);
    compare(Ac, Ac_rap);
    delete Ac_rap;
    delete Ac;
    delete P_csc;
    delete AP;
    delete P;
    delete A;

    // TEST LEVEL 1
    A = readParMatrix((char *)"../../../../test_data/rss_A1.mtx", MPI_COMM_WORLD, 1, 0);
    P = readParMatrix((char *)"../../../../test_data/rss_P1.mtx", MPI_COMM_WORLD, 1, 0);
    AP = A->mult(P);
    P_csc = new ParCSCMatrix(P);
    Ac = AP->mult_T(P_csc);
    Ac_rap = readParMatrix((char *)"../../../../test_data/rss_A2.mtx", MPI_COMM_WORLD, 1, 0);
    compare(Ac, Ac_rap);
    delete Ac_rap;
    delete Ac;
    delete P_csc;
    delete AP;
    delete P;
    delete A;
} // end of TEST(TestParRAP, TestsInRuge_Stuben) //
