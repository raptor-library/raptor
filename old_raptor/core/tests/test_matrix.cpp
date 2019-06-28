// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"
using namespace raptor;


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

} // end of main() //

TEST(MatrixTest, TestsInCore)
{
    int rows[10] = {22, 17, 12, 0, 5, 7, 1, 0, 0, 12};
    int cols[10] = {5, 18, 21, 0, 7, 7, 0, 1, 0, 21};
    double vals[10] = {2.0, 1.0, 0.5, 1.0, 2.0, 1.0, 1.2, 2.2, 1.5, -1.0};

    int row_ctr[26] = {0, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 9, 
        9, 9, 9, 9, 10, 10, 10};

    // Create COO Matrix (25x25)
    COOMatrix* A_coo = new COOMatrix(25, 25, 1);
    for (int i = 0; i < 10; i++)
    {
        A_coo->add_value(rows[i], cols[i], vals[i]);
    }

    // Check dimensions of A_coo
    ASSERT_EQ(A_coo->n_rows, 25);
    ASSERT_EQ(A_coo->n_cols, 25);
    ASSERT_EQ(A_coo->nnz, 10);

   // Check that rows, columns, and values in A_coo are correct
    for (int i = 0; i < 10; i++)
    {
        ASSERT_EQ(A_coo->idx1[i], rows[i]);
        ASSERT_EQ(A_coo->idx2[i], cols[i]);
        ASSERT_EQ(A_coo->vals[i], vals[i]);
    }

    // Create CSR Matrix from COO
    CSRMatrix* A_csr = A_coo->to_CSR();

    // Check dimensions of A_csr
    ASSERT_EQ(A_csr->n_rows,25);
    ASSERT_EQ(A_csr->n_cols,25);
    ASSERT_EQ(A_csr->nnz,10);

    // Check that rows, columns, and values in A_coo are correct
    
    for (int i = 0; i < 26; i++)
    {
        ASSERT_EQ(A_csr->idx1[i],row_ctr[i]);
    }

    delete A_coo;
    delete A_csr;

} // end of TEST(MatrixTest, TestsInCore) //

