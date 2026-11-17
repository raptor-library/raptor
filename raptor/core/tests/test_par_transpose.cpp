// SPDX-FileCopyrightText: RAPtor Core Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.

#include "gtest/gtest.h"
#include "raptor/raptor.hpp"
#include "raptor/tests/par_compare.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp=RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;

} // end of main() //

TEST(ParMatrixTest, TestsInCore)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    ParCSRMatrix* A = readParMatrix(RAPTOR_TEST_DATA_DIR "aniso.pm");
    ParCSRMatrix* AT_py = readParMatrix(RAPTOR_TEST_DATA_DIR "aniso_T.pm");
    ParCSRMatrix* AT = (ParCSRMatrix*) A->transpose();
    A->sort();
    AT->sort();
    AT_py->sort();
    compare(AT, AT_py);
    delete A;
    delete AT_py;
    delete AT;

    A = readParMatrix(RAPTOR_TEST_DATA_DIR "laplacian.pm");
    AT_py = readParMatrix(RAPTOR_TEST_DATA_DIR "laplacian_T.pm");
    AT = (ParCSRMatrix*) A->transpose();
    A->sort();
    AT->sort();
    AT_py->sort();
    compare(AT, AT_py);
    delete A;
    delete AT_py;
    delete AT;


} // end of TEST(ParMatrixTest, TestsInCore) //

