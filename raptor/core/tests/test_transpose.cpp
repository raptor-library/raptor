// SPDX-FileCopyrightText: RAPtor Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.

#include "gtest/gtest.h"
#include "raptor/raptor.hpp"
#include "raptor/tests/compare.hpp"

using namespace raptor;


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

} // end of main() //

TEST(MatrixTest, TestsInCore)
{
    CSRMatrix* A = readMatrix("../../../../test_data/aniso.pm");
    CSRMatrix* AT_py = readMatrix("../../../../test_data/aniso_T.pm");
    CSRMatrix* AT = (CSRMatrix*) A->transpose();
    A->sort();
    AT->sort();
    AT_py->sort();
    compare(AT, AT_py);
    delete A;
    delete AT_py;
    delete AT;

    A = readMatrix("../../../../test_data/laplacian.pm");
    AT_py = readMatrix("../../../../test_data/laplacian_T.pm");
    AT = (CSRMatrix*) A->transpose();
    A->sort();
    AT->sort();
    AT_py->sort();
    compare(AT, AT_py);
    delete A;
    delete AT_py;
    delete AT;


} // end of TEST(MatrixTest, TestsInCore) //


