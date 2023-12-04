// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "gtest/gtest.h"
#include "raptor/raptor.hpp"
#include "raptor/tests/par_compare.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp = RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //

TEST(ParMatrixAddTest, TestsInUtil)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    ParCSRMatrix* A;
    ParCSRMatrix* S;
    ParCSRMatrix* AS;
    ParCSRMatrix* AS_rap;

    const char* A0_fn = "../../../../test_data/rss_A0.pm";
    const char* S0_fn = "../../../../test_data/rss_S0.pm";
    const char* AS0_fn = "../../../../test_data/rss_AS.pm";
    const char* AmS0_fn = "../../../../test_data/rss_AmS.pm";

    // TEST LEVEL 0
    A = readParMatrix(A0_fn);
    S = readParMatrix(S0_fn);
    AS = readParMatrix(AS0_fn);
    AS_rap = A->add(S);

    compare(AS, AS_rap);
    delete AS;
    delete AS_rap;

    AS = readParMatrix(AmS0_fn);
    AS_rap = A->subtract(S);

    compare(AS, AS_rap);

    delete AS_rap;
    delete AS;
    delete S;
    delete A;


 
} // end of TEST(ParMatrixAddTest, TestsInUtil) //

