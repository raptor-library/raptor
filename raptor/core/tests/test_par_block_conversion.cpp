// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "raptor.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp=RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;

} // end of main() //

TEST(ParBlockMatrixTest, TestsInCore)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    setenv("PPN", "4", 1);

    int start, end, pos;
    int col, prev_col, block_col;
    int prev_row, block_row;
    int block_pos, row_pos, col_pos;
    int global_col;
    double val;

    // Form standard anisotropic matrix
    double eps = 0.001;
    double theta = M_PI / 8.0;
    int block_n = 2;
    std::vector<int> grid(2, num_procs*block_n);
    //std::vector<int> grid(2, 4);
    double* stencil = diffusion_stencil_2d(eps, theta);
    CSRMatrix* A_serial = stencil_grid(stencil, grid.data(), 2);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid.data(), 2);

    ParBSRMatrix* A_bsr = A->to_ParBSR(block_n, block_n);
    ParCSRMatrix* A_csr_from_bsr = A_bsr->to_ParCSR();
    
    int lcl_nnz = A_csr_from_bsr->local_nnz;
    int nnz;
    MPI_Allreduce(&lcl_nnz, &nnz, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    ASSERT_EQ(A_serial->nnz,nnz);

    /*for (int i = 0; i < num_procs; i++)
    {
        if (i == rank)
        {
            for (int j = 0; j < A_csr_from_bsr->partition->first_cols.size(); j++)
            {
                printf("%d ", A_csr_from_bsr->partition->first_cols[j]);
            }
            printf("\n");
            fflush(stdout);   
            for (int j = 0; j < A->partition->first_cols.size(); j++)
            {
                printf("%d ", A->partition->first_cols[j]);
            }
            printf("\n");
            fflush(stdout);   
            for (int j = 0; j < A_bsr->partition->first_cols.size(); j++)
            {
                printf("%d ", A_bsr->partition->first_cols[j]);
            }
            printf("\n");
            fflush(stdout);   
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }*/

    ASSERT_EQ(A->partition->local_num_rows, A_csr_from_bsr->partition->local_num_rows);
    ASSERT_EQ(A->partition->local_num_cols, A_csr_from_bsr->partition->local_num_cols);
    ASSERT_EQ(A->partition->first_local_row, A_csr_from_bsr->partition->first_local_row);
    ASSERT_EQ(A->partition->first_local_col, A_csr_from_bsr->partition->first_local_col);
    ASSERT_EQ(A->partition->last_local_row, A_csr_from_bsr->partition->last_local_row);
    ASSERT_EQ(A->partition->last_local_col, A_csr_from_bsr->partition->last_local_col);
    
    ParVector x(A->global_num_rows, A->local_num_rows);
    ParVector b(A->global_num_rows, A->local_num_rows);
    ParVector tmp(A->global_num_rows, A->local_num_rows);
    x.set_const_value(1.0);

    // Test BSR to CSR SpMV
    A_bsr->mult(x, b);
    A_csr_from_bsr->mult(x, tmp);
    for (int i = 0; i < A_bsr->local_num_rows*block_n; i++)
        ASSERT_NEAR(tmp[i], b[i], 1e-10);

    // Test BSR to CSR Transpose SpMV
    /*A_bsr->mult_T(x, b);
    A_csr_from_bsr->mult_T(x, tmp);
    for (int i = 0; i < A_bsr->local_num_rows*block_n; i++)
        ASSERT_NEAR(tmp[i], b[i], 1e-10);*/

    // Test BSR to CSR TAPSpMVs 
    A_bsr->tap_mult(x, b);
    A_csr_from_bsr->tap_mult(x, tmp);
    for (int i = 0; i < A_bsr->local_num_rows*block_n; i++)
        ASSERT_NEAR(tmp[i], b[i], 1e-10);

    // Test BSR to CSR Transpose TAPSpMV
    /*A_bsr->tap_mult_T(x, b);
    A_csr_from_bsr->tap_mult_T(x, tmp);
    for (int i = 0; i < A_bsr->local_num_rows*block_n; i++)
        ASSERT_NEAR(tmp[i], b[i], 1e-10);*/

    delete A;
    delete A_bsr;
    delete A_csr_from_bsr;

    setenv("PPN", "16", 1);
    

} // end of TEST(MatrixTest, TestsInCore) //



