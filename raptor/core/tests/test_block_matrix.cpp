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

TEST(BlockMatrixTest, TestsInCore)
{
/*
    Matrix* A_coo_mat;
    Matrix* A_bcoo_mat;
    COOMatrix* A_coo;
    COOMatrix* A_bcoo_coo;
    BCOOMatrix* A_bcoo;

    int num_rows, num_cols;
    int block_rows, block_cols;
    int row_block_size, col_block_size;
    int block_size;
    aligned_vector<double> vals;


    // Test initialization from dense
    row_block_size = 2;
    col_block_size = 2;
    block_rows = 5;
    block_cols = 5;
    vals.resize(100);
    std::iota(vals.begin(), vals.end(), 1);
    A_coo = new COOMatrix(block_rows * row_block_size, block_cols * col_block_size, vals.data());
    A_bcoo = new BCOOMatrix(block_rows, block_cols, row_block_size, col_block_size, vals.data());
    A_coo->sort();
    A_bcoo->sort();
    ASSERT_EQ(A_coo->nnz, A_bcoo->nnz * A_bcoo->b_size);
    ASSERT_EQ(A_coo->n_rows, A_bcoo->n_rows * A_bcoo->b_rows);
    ASSERT_EQ(A_coo->n_cols, A_bcoo->n_cols * A_bcoo->b_cols);
    for (int i = 0; i < A_bcoo->nnz; i++)
    {
        int block_row = A_bcoo->idx1[i];
        int block_col = A_bcoo->idx2[i];
        int first_row = block_row * A_bcoo->b_rows; 
        int first_col = block_col * A_bcoo->b_cols;
        for (int j = 0; j < A_bcoo->b_rows; j++)
        {
            for (int k = 0; k < A_bcoo->b_cols; k++)
            {
                int block_idx = j*A_bcoo->b_cols + k;
                int point_idx = ((first_row + j) * A_coo->n_cols) + first_col + k;
                ASSERT_EQ(block_row * A_bcoo->b_rows + j, A_coo->idx1[point_idx]);
                ASSERT_EQ(block_col * A_bcoo->b_cols + k, A_coo->idx2[point_idx]);
                ASSERT_NEAR(A_coo->vals[point_idx], A_bcoo->vals[i][block_idx], 1e-06);
            }
        }
    }
    delete A_coo;
    delete A_bcoo;

    
    // Create new COO and BCOO matrices, with
    // different declaration types
    int num_block_rows = 10;
    int num_block_cols = 10;
    int block_nnz = 5;
    int brows[5] = {2, 5, 3, 7, 2};
    int bcols[5] = {3, 6, 3, 1, 2};
    double bvals[20] = {10.2, 12.3, 15.2, 43., 56, 3.2, 5.3, 1.4, 14.6, 23.,
        10, 12.3, 51, 27.6, 2, 7, 37.7, 12.2, 23, 1.2}; 
    row_block_size = 2;
    col_block_size = 2;
    block_size = row_block_size * col_block_size;
    num_rows = num_block_rows * row_block_size;
    num_cols = num_block_cols * col_block_size;

    A_coo_mat = new COOMatrix(num_rows, num_cols);
    A_bcoo_mat = new BCOOMatrix(num_block_rows, num_block_cols, row_block_size, 
            col_block_size);
    A_coo = new COOMatrix(num_rows, num_cols);
    A_bcoo_coo = new BCOOMatrix(num_block_rows, num_block_cols, row_block_size, 
            col_block_size);
    A_bcoo = new BCOOMatrix(num_block_rows, num_block_cols, row_block_size, 
            col_block_size);

    // Check correctness of format
    ASSERT_EQ(A_coo_mat->format(), COO);
    ASSERT_EQ(A_coo->format(), COO);
    ASSERT_EQ(A_bcoo_mat->format(), BCOO);
    ASSERT_EQ(A_bcoo_coo->format(), BCOO);
    ASSERT_EQ(A_bcoo->format(), BCOO);

    // Test Add_Value method
    for (int i = 0; i < block_nnz; i++)
    {
        int block_row = brows[i];
        int block_col = bcols[i];
        A_bcoo->add_value(block_row, block_col, &bvals[i*block_size]);
        for (int j = 0; j < row_block_size; j++)
        {
            for (int k = 0; k < col_block_size; k++)
            {
                A_coo->add_value(block_row * num_block_rows + j, 
                        block_col * num_block_cols + k,
                        &bvals[i + j*col_block_size + k]);
            }
        }
    }


    delete A_coo_mat;
    delete A_bcoo_mat;
    delete A_coo;
    delete A_bcoo_coo;
    delete A_bcoo;

*/

} // end of TEST(MatrixTest, TestsInCore) //


