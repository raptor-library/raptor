// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "raptor.hpp"
using namespace raptor;

void compare_vals(CSRMatrix* A, BSRMatrix* B)
{
    A->sort();
    B->sort();
    int ctr = 0;
    for (int i = 0; i < B->n_rows; i++)
    {
        for (int k = 0; k < B->b_rows; k++)
        {
            for (int j = B->idx1[i]; j < B->idx1[i+1]; j++)
            {
                int b_col = B->idx2[j];
                double* val = B->block_vals[j];
                for (int l = 0; l < B->b_cols; l++)
                {
                    if (fabs(val[k*B->b_cols + l]) > zero_tol)
                        ASSERT_NEAR(val[k*B->b_cols + l], A->vals[ctr++], 1e-10);

                }
            }
        }
    }

}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

} // end of main() //

TEST(BlockMatrixTest, TestsInCore)
{
    int block_row_size = 2;
    int block_col_size = 2;
    int block_size = 4;
    int block_nnz = 5;
    int block_num_rows = 3;
    int block_num_cols = 3;
    int num_rows = block_num_rows * block_row_size;
    int num_cols = block_num_cols * block_col_size;
    int nnz = block_nnz * block_size;

    aligned_vector<int> rows = {0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 5, 5};
    aligned_vector<int> cols = {0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 4, 5, 4, 5};
    aligned_vector<double> vals = {1.0, 0.0, 2.0, 1.0, 6.0, 7.0, 8.0, 2.0, 1.0, 4.0, 5.0, 1.0,
                                4.0, 3.0, 0.0, 0.0, 7.0, 2.0, 0.0, 0.0};

    aligned_vector<int> block_row_ptr = {0,2,3,5};    
    aligned_vector<int> block_rows = {0, 0, 1, 2, 2};
    aligned_vector<int> block_cols = {0, 1, 1, 1, 2};
    aligned_vector<double*> block_vals;
    for (int i = 0; i < block_nnz; i++)
    {
        double* block = new double[block_size];
        for (int j = 0; j < block_size; j++)
        {
            block[j] = vals[i*block_size+j];
        }
        block_vals.push_back(block);
    }

    Matrix* A_bcoo = new BCOOMatrix(block_num_rows, block_num_cols,
            block_row_size, block_col_size);
    for (int i = 0; i < block_nnz; i++)
        A_bcoo->add_value(block_rows[i], block_cols[i], block_vals[i]);

    Matrix* A_coo = new COOMatrix(num_rows, num_cols);
    for (int i = 0; i < nnz; i++)
        A_coo->add_value(rows[i], cols[i], vals[i]);

    Matrix* A_bsr = A_bcoo->to_CSR();
    Matrix* A_csr = A_coo->to_CSR();
    Matrix* A_bsc = A_bsr->to_CSC();
    Matrix* A_csc = A_csr->to_CSC();
    Matrix* A_csr_from_bsr = A_bsr->to_CSR();

    Vector x(num_rows);
    Vector b(num_cols);
    Vector tmp(num_cols);
    x.set_const_value(1.0);

    A_bcoo->sort();
    A_bcoo->move_diag();
    A_bcoo->remove_duplicates();

    A_bsr->sort();
    A_bsr->move_diag();
    A_bsr->remove_duplicates();

    A_bsc->sort();
    A_bsc->move_diag();
    A_bsc->remove_duplicates();

    ASSERT_EQ(A_bcoo->n_rows, A_bsr->n_rows);
    ASSERT_EQ(A_bsr->n_rows, A_bsc->n_rows);
    ASSERT_EQ(A_bcoo->n_cols, A_bsr->n_cols);
    ASSERT_EQ(A_bsr->n_cols, A_bsc->n_cols);
    ASSERT_EQ(A_bcoo->nnz, A_bsr->nnz);
    ASSERT_EQ(A_bsr->nnz, A_bsc->nnz);
    ASSERT_EQ(A_csr_from_bsr->nnz, A_csr->nnz);

    double** bcoo_vals = (double**) A_bcoo->get_data();
    double** bsr_vals = (double**) A_bsr->get_data();
    for (int i = 0; i < A_bcoo->nnz; i++)
    {
        for (int j = 0; j < A_bcoo->b_size; j++)
        {
            ASSERT_NEAR(bcoo_vals[i][j], bsr_vals[i][j], 1e-10);
        }
    }

    Matrix* Atmp = A_bsc->to_CSR();
    Atmp->sort();
    Atmp->move_diag();
    double** tmp_vals = (double**) Atmp->get_data();
    for (int i = 0; i < A_bsr->nnz; i++)
    {
        for (int j = 0; j < A_bsr->b_size; j++)
        {
            ASSERT_NEAR(bsr_vals[i][j], tmp_vals[i][j], 1e-10);
        }
    }

    ASSERT_EQ(A_bcoo->format(), BCOO);
    ASSERT_EQ(A_coo->format(), COO);
    ASSERT_EQ(A_bsr->format(), BSR);
    ASSERT_EQ(A_csr->format(), CSR);
    ASSERT_EQ(A_bsc->format(), BSC);
    ASSERT_EQ(A_csc->format(), CSC);
    ASSERT_EQ(A_csr_from_bsr->format(), CSR);

    A_csr->mult(x, b);
    A_bsr->mult(x, tmp);
    for (int i = 0; i < num_cols; i++)
        ASSERT_NEAR(b[i], tmp[i], 1e-10);
    
    A_csr_from_bsr->mult(x, tmp);
    for (int i = 0; i < num_cols; i++)
        ASSERT_NEAR(b[i], tmp[i], 1e-10);

    A_coo->mult(x, tmp);
    for (int i = 0; i < num_cols; i++)
        ASSERT_NEAR(b[i], tmp[i], 1e-10);

    A_bcoo->mult(x, tmp);
    for (int i = 0; i < num_cols; i++)
        ASSERT_NEAR(b[i], tmp[i], 1e-10);

    A_csc->mult(x, tmp);
    for (int i = 0; i < num_cols; i++)
        ASSERT_NEAR(b[i], tmp[i], 1e-10);

    A_bsc->mult(x, tmp);
    for (int i = 0; i < num_cols; i++)
        ASSERT_NEAR(b[i], tmp[i], 1e-10);


    CSRMatrix* C_csr = A_csr->mult((CSRMatrix*)A_csr);
    CSRMatrix* C_bsr = A_bsr->mult((BSRMatrix*)A_bsr);
    ASSERT_EQ(C_csr->n_rows, C_bsr->n_rows * C_bsr->b_rows);
    ASSERT_EQ(C_csr->n_cols, C_bsr->n_cols * C_bsr->b_cols);
    compare_vals(C_csr, (BSRMatrix*) C_bsr);

    CSRMatrix* C_csr_from_bsr = A_csr_from_bsr->mult((CSRMatrix*)A_csr_from_bsr);
    ASSERT_EQ(C_csr_from_bsr->n_rows, C_bsr->n_rows * C_bsr->b_rows);
    ASSERT_EQ(C_csr_from_bsr->n_cols, C_bsr->n_cols * C_bsr->b_cols);
    compare_vals(C_csr_from_bsr, (BSRMatrix*) C_bsr);

    CSRMatrix* D_csr = A_csr->mult_T((CSCMatrix*)A_csc);
    CSRMatrix* D_bsr = A_bsr->mult_T((BSCMatrix*)A_bsc);
    ASSERT_EQ(D_csr->n_rows, D_bsr->n_rows * D_bsr->b_rows);
    ASSERT_EQ(D_csr->n_cols, D_bsr->n_cols * D_bsr->b_cols);
    compare_vals(D_csr, (BSRMatrix*) D_bsr);
    
    CSRMatrix* D_csr_from_bsr = A_csr_from_bsr->mult_T((CSCMatrix*)A_csc);
    ASSERT_EQ(D_csr_from_bsr->n_rows, D_bsr->n_rows * D_bsr->b_rows);
    ASSERT_EQ(D_csr_from_bsr->n_cols, D_bsr->n_cols * D_bsr->b_cols);
    compare_vals(D_csr_from_bsr, (BSRMatrix*) D_bsr);

    delete A_bsr;
    delete A_csr;
    delete A_bsc;
    delete A_csc;
    delete A_bcoo;
    delete A_coo;
    delete A_csr_from_bsr;

    delete C_csr;
    delete C_bsr;
    delete C_csr_from_bsr;

    delete D_csr;
    delete D_bsr;
    delete D_csr_from_bsr;
    
    for (aligned_vector<double*>::iterator it = block_vals.begin();
            it != block_vals.end(); ++it)
        delete[] *it;

} // end of TEST(MatrixTest, TestsInCore) //


