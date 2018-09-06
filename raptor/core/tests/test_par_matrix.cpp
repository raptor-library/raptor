// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/par_matrix.hpp"
#include "gallery/stencil.hpp"
#include "gallery/par_stencil.hpp"
#include "gallery/diffusion.hpp"

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

    double eps = 0.001;
    double theta = M_PI / 8.0;
    int grid[2] = {10, 10};
    double* stencil = diffusion_stencil_2d(eps, theta);
    CSRMatrix* A = stencil_grid(stencil, grid, 2);
    ParCSRMatrix* A_par = par_stencil_grid(stencil, grid, 2);

    ParCSCMatrix* A_par_csc = A_par->to_ParCSC();

    int lcl_nnz = A_par->local_nnz;
    int nnz;
    MPI_Allreduce(&lcl_nnz, &nnz, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    ASSERT_EQ(A->nnz,nnz);

    double A_dense[10000] = {0};
    for (int i = 0; i < A->n_rows; i++)
    {
        for (int j = A->idx1[i]; j < A->idx1[i+1]; j++)
        {
            A_dense[i*100 + A->idx2[j]] = A->vals[j];
        }
    }

    // Compare A_par to A_dense
    for (int i = 0; i < A_par->local_num_rows; i++)
    {
        int row = A_par->local_row_map[i];
        for (int j = A_par->on_proc->idx1[i]; j < A_par->on_proc->idx1[i+1]; j++)
        {
            int col = A_par->on_proc_column_map[A_par->on_proc->idx2[j]];
            //ASSERT_LT((fabs(A_dense[row*100+col] - A_par->on_proc->vals[j])), zero_tol);
            ASSERT_NEAR(A_dense[row*100+col], A_par->on_proc->vals[j], zero_tol);
        }

        for (int j = A_par->off_proc->idx1[i]; j < A_par->off_proc->idx1[i+1]; j++)
        {
            int col = A_par->off_proc_column_map[A_par->off_proc->idx2[j]];
            ASSERT_NEAR(A_dense[row*100+col], A_par->off_proc->vals[j], zero_tol);
        }
    }

    // Compare A_par_csc to A_dense
    for (int i = 0; i < A_par_csc->on_proc_num_cols; i++)
    {
        int col = A_par_csc->on_proc_column_map[i];
        for (int j = A_par_csc->on_proc->idx1[i]; j < A_par_csc->on_proc->idx1[i+1]; j++)
        {
            int row = A_par_csc->local_row_map[A_par_csc->on_proc->idx2[j]];
            ASSERT_NEAR(A_dense[row*100+col],A_par_csc->on_proc->vals[j], zero_tol);
        }
    }

    for (int i = 0; i < A_par_csc->off_proc_num_cols; i++)
    {
        int col = A_par_csc->off_proc_column_map[i];
        for (int j = A_par_csc->off_proc->idx1[i]; j < A_par_csc->off_proc->idx1[i+1]; j++)
        {
            int row = A_par_csc->local_row_map[A_par_csc->off_proc->idx2[j]];
            ASSERT_NEAR(A_dense[row*100+col], A_par_csc->off_proc->vals[j], zero_tol);
        }
    }

    delete[] stencil;

} // end of TEST(ParMatrixTest, TestsInCore) //
