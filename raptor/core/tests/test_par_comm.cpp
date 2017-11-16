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

TEST(ParCommTest, TestsInCore)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double eps = 0.001;
    double theta = M_PI / 8.0;
    int grid[2] = {10, 10};
    int global_row, global_col;
    int start, end;
    double val;
    double* stencil = diffusion_stencil_2d(eps, theta);
    std::vector<int> sendbuf;
    std::vector<double> seq_row;

    CSRMatrix* A_seq = stencil_grid(stencil, grid, 2);

    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);

    ParVector x(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    if (A->local_num_rows)
    {
        sendbuf.resize(A->local_num_rows);
        for (int i = 0; i < A->local_num_rows; i++)
        {
            sendbuf[i] = A->local_row_map[i];
        }
    }

    A->comm->communicate(sendbuf);

    ASSERT_GT(A->off_proc_num_cols, 0);
    for (int i = 0; i < A->off_proc_num_cols; i++)
    {
        ASSERT_EQ(A->comm->recv_data->int_buffer[i], A->off_proc_column_map[i]);
    }

    seq_row.resize(A_seq->n_cols);
    CSRMatrix* recv_mat = A->comm->communicate(A);
    for (int i = 0; i < A->off_proc_num_cols; i++)
    {
        global_row = A->off_proc_column_map[i];
        start = A_seq->idx1[global_row];
        end = A_seq->idx1[global_row+1];
        for (int j = start; j < end; j++)
        {
            seq_row[A_seq->idx2[j]] = A_seq->vals[j];
        }

        start = recv_mat->idx1[i];
        end = recv_mat->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            global_col = recv_mat->idx2[j];
            val = recv_mat->vals[j];
            ASSERT_NEAR(seq_row[global_col], val, 1e-06);
        }
    }

    delete A;
    delete recv_mat;

} // end of TEST(ParCommTest, TestsInCore) //
