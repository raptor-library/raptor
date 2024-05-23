// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "raptor/raptor.hpp"
using namespace raptor;


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

} // end of main() //

TEST(BSRMatrixTest, TestsInCore)
{
    // Matrix [0, 1], [1, 0]
    //        [2, 0], [0, 2]
    //        [3, 0], [0, 0]
    //        [0, 4], [0, 0]
    int n_csr = 4;
    int nnz_csr = 6;
    std::vector<int> rowptr_csr = {0, 2, 4, 5, 6};
    std::vector<int> col_idx_csr = {1, 2, 0, 3, 0, 1};
    std::vector<double> data_csr = {1, 1, 2, 2, 3, 4};
    CSRMatrix* A_csr = new CSRMatrix(n_csr, n_csr, rowptr_csr, col_idx_csr, data_csr);
    
    int n = 2;  // 2 blocks by 2 blocks
    int br = 2; // blocks are each 2x2
    int bs = 4; 
    int nnz = 3; // 3 blocks
    std::vector<int> rowptr = {0, 2, 3};
    std::vector<int> col_idx = {0, 1, 0};
    std::vector<double> data = {0, 1, 2, 0, 1, 0, 0, 2, 3, 0, 0, 4};

    // Hardcode one BSR Matrix
    BSRMatrix* A = new BSRMatrix(n, n, br, br, nnz);
    A->idx1[0] = 0;
    for (int i = 0; i < n; i++)
    {
        A->idx1[i+1] = rowptr[i+1];
        for (int j = A->idx1[i]; j < A->idx1[i+1]; j++)
        {
            A->idx2.push_back(col_idx[j]);
            double* vals = new double[bs];
            for (int k = 0; k < bs; k++)
                vals[k] = data[j*bs + k];
            A->block_vals.push_back(vals);
        }
    }

    // Call method that converts CSR to BSR
    BSRMatrix* A_conv = new BSRMatrix(A_csr, br, br);

    // Check that both BSR matrices are equivalent
    ASSERT_EQ(A_conv->n_rows, A->n_rows);
    ASSERT_EQ(A_conv->n_cols, A->n_cols);
    ASSERT_EQ(A_conv->b_rows, A->b_rows);
    ASSERT_EQ(A_conv->b_cols, A->b_cols);
    ASSERT_EQ(A_conv->b_size, A->b_size);

    for (int i = 0; i < A->n_rows; i++)
    {
        ASSERT_EQ(A_conv->idx1[i+1], A->idx1[i+1]);
        for (int j = A->idx1[i]; j < A->idx1[i+1]; j++)
        {
            ASSERT_EQ(A_conv->idx2[j], A->idx2[j]);
            for (int k = 0; k < A->b_size; k++)
                ASSERT_EQ(A_conv->block_vals[j][k], A->block_vals[j][k]);
        }
    }
                 
    delete A_csr;
    delete A;
    delete A_conv;

} // end of TEST(MatrixTest, TestsInCore) //

