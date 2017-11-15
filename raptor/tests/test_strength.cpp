// EXPECT_EQ and ASSERT_EQ are macros
// EXPECT_EQ test execution and continues even if there is a failure
// ASSERT_EQ test execution and aborts if there is a failure
// The ASSERT_* variants abort the program execution if an assertion fails 
// while EXPECT_* variants continue with the run.

#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "gallery/matrix_IO.hpp"
#include "tests/compare.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

} // end of main() //

TEST(StrengthTest, TestsIntests)
{
    CSRMatrix* A;
    CSRMatrix* S;
    CSRMatrix* S_rap;

    const char* A0_fn = "../../../test_data/rss_A0.mtx";
    const char* A1_fn = "../../../test_data/rss_A1.mtx";
    const char* S0_fn = "../../../test_data/rss_S0.mtx";
    const char* S1_fn = "../../../test_data/rss_S1.mtx";

    A = readMatrix(A0_fn, 1);
    S = readMatrix(S0_fn, 1);
    S_rap = A->strength(0.25);
    compare_pattern(S, S_rap);
    delete A;
    delete S;
    delete S_rap;

    A = readMatrix(A1_fn, 0);
    S = readMatrix(S1_fn, 0);
    S_rap = A->strength(0.25);
    compare_pattern(S, S_rap);
    delete A;
    delete S;
    delete S_rap;

} // end of TEST(StrengthTest, TestsIntests) //
