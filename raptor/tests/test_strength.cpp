// EXPECT_EQ and ASSERT_EQ are macros
// EXPECT_EQ test execution and continues even if there is a failure
// ASSERT_EQ test execution and aborts if there is a failure
// The ASSERT_* variants abort the program execution if an assertion fails 
// while EXPECT_* variants continue with the run.

#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "gallery/matrix_IO.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

} // end of main() //


void compare(CSRMatrix* S, CSRMatrix* S_rap)
{
    int start, end;

    S->sort();
    S->move_diag();
    S_rap->sort();
    S_rap->move_diag();

    ASSERT_EQ(S->n_rows, S_rap->n_rows);
    ASSERT_EQ(S->n_cols, S_rap->n_cols);
    ASSERT_EQ(S->nnz, S_rap->nnz);

    ASSERT_EQ(S->idx1[0], S_rap->idx1[0]);
    for (int i = 0; i < S->n_rows; i++)
    {
        ASSERT_EQ(S->idx1[i+1], S_rap->idx1[i+1]);
        start = S->idx1[i];
        end = S->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            ASSERT_EQ(S->idx2[j], S_rap->idx2[j]);
            ASSERT_NEAR(S->vals[j],S_rap->vals[j], 1e-06);
        }
    }
}

TEST(StrengthTest, TestsIntests)
{
    CSRMatrix* A;
    CSRMatrix* S;
    CSRMatrix* S_rap;

    A = readMatrix("rss_laplace_A0.mtx", 1);
    S = readMatrix("rss_laplace_S0.mtx", 1);
    S_rap = A->strength(0.25);
    compare(S, S_rap);
    delete A;
    delete S;
    delete S_rap;

    A = readMatrix("rss_laplace_A1.mtx", 0);
    S = readMatrix("rss_laplace_S1.mtx", 0);
    S_rap = A->strength(0.25);
    compare(S, S_rap);
    delete A;
    delete S;
    delete S_rap;

    A = readMatrix("rss_aniso_A0.mtx", 1);
    S = readMatrix("rss_aniso_S0.mtx", 1);
    S_rap = A->strength();
    compare(S, S_rap);
    delete A;
    delete S;
    delete S_rap;

    A = readMatrix("rss_aniso_A1.mtx", 0);
    S = readMatrix("rss_aniso_S1.mtx", 0);
    S_rap = A->strength();
    compare(S, S_rap);
    delete A;
    delete S;
    delete S_rap;

} // end of TEST(StrengthTest, TestsIntests) //
