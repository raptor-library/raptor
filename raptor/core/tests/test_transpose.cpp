// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "gallery/matrix_IO.hpp"
#include "tests/compare.hpp"

using namespace raptor;


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

} // end of main() //

TEST(MatrixTest, TestsInCore)
{
    CSRMatrix* A = readMatrix("../../../../test_data/standard.pm");
    CSRMatrix* AT_py = readMatrix("../../../../test_data/transpose.pm");
    CSRMatrix* AT = (CSRMatrix*) A->transpose();

    A->sort();
    AT->sort();
    AT_py->sort();
    compare(AT, AT_py);

} // end of TEST(MatrixTest, TestsInCore) //


