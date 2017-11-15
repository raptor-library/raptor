// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/par_stencil.hpp"
#include "gallery/par_matrix_IO.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp = RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //

TEST(ParAnisoTest, TestsInGallery)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n_rows, n_cols; 
    int global_n_rows, global_n_cols;
    int row, row_nnz, nnz;
    int start, end;
    double row_sum, sum;
    int grid[2] = {25, 25};
    double eps = 0.001;
    double theta = M_PI/8.0;
    double* stencil = diffusion_stencil_2d(eps, theta);
    ParCSRMatrix* A_sten = par_stencil_grid(stencil, grid, 2);
    ParCSRMatrix* A_io = readParMatrix("../../../../test_data/aniso.mtx");

    // Open laplacian data file
    FILE *f = fopen("../../../../test_data/aniso_data.txt", "r");

    // Read global shape
    fscanf(f, "%d %d\n", &n_rows, &n_cols);

    // Compare shapes
    ASSERT_EQ(n_rows, A_sten->global_num_rows);
    ASSERT_EQ(n_cols, A_sten->global_num_cols);
    ASSERT_EQ(n_rows, A_io->global_num_rows);
    ASSERT_EQ(n_cols, A_io->global_num_cols);

    ASSERT_EQ(A_sten->local_num_rows, A_io->local_num_rows);
    ASSERT_EQ(A_sten->on_proc_num_cols, A_io->on_proc_num_cols);
    ASSERT_EQ(A_sten->partition->first_local_row, A_io->partition->first_local_row);
    ASSERT_EQ(A_sten->partition->last_local_row, A_io->partition->last_local_row);
    ASSERT_EQ(A_sten->partition->first_local_col, A_io->partition->first_local_col);
    ASSERT_EQ(A_sten->partition->last_local_col, A_io->partition->last_local_col);

    MPI_Allreduce(&A_sten->local_num_rows, &global_n_rows, 1, MPI_INT,
            MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&A_sten->on_proc_num_cols, &global_n_cols, 1, MPI_INT,
            MPI_SUM, MPI_COMM_WORLD);
    ASSERT_EQ(global_n_rows, n_rows);
    ASSERT_EQ(global_n_cols, n_cols);

    std::vector<int> global_col_starts(num_procs+1);
    std::vector<int> global_row_starts(num_procs+1);
    MPI_Allgather(&A_sten->partition->first_local_row, 1, MPI_INT, &global_row_starts[0],
            1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&A_sten->partition->first_local_col, 1, MPI_INT, &global_col_starts[0],
            1, MPI_INT, MPI_COMM_WORLD);
    global_row_starts[num_procs] = A_sten->global_num_rows;
    global_col_starts[num_procs] = A_sten->global_num_cols;

    ASSERT_EQ(A_sten->local_num_rows, (global_row_starts[rank+1] - global_row_starts[rank]));
    ASSERT_EQ(A_sten->on_proc_num_cols, (global_col_starts[rank+1] - global_col_starts[rank]));

    if (A_sten->local_num_rows)
    {
        ASSERT_EQ(A_sten->partition->last_local_row, (global_col_starts[rank+1] - 1));
    }
    if (A_sten->on_proc_num_cols)
    {
        ASSERT_EQ(A_sten->partition->last_local_col, (global_col_starts[rank+1] - 1));
    }

    A_sten->sort();
    A_io->sort();

    ASSERT_EQ(A_sten->on_proc->idx1[0], A_io->on_proc->idx1[0]);
    ASSERT_EQ(A_sten->off_proc->idx1[0], A_io->off_proc->idx1[0]);

    for (int i = 0; i < A_sten->partition->first_local_row; i++)
    {
        fscanf(f, "%d %d %lg\n", &row, &row_nnz, &row_sum);
    }

    for (int i = 0; i < A_sten->local_num_rows; i++)
    {
        fscanf(f, "%d %d %lg\n", &row, &row_nnz, &row_sum);

        ASSERT_EQ(A_sten->on_proc->idx1[i+1], A_io->on_proc->idx1[i+1]);
        start = A_sten->on_proc->idx1[i];
        end = A_sten->on_proc->idx1[i+1];
        nnz = end - start;
        sum = 0.0;

        for (int j = start; j < end; j++)
        {
            ASSERT_EQ(A_sten->on_proc->idx2[j], A_io->on_proc->idx2[j]);
            ASSERT_NEAR(A_sten->on_proc->vals[j], A_io->on_proc->vals[j], 1e-05);
            sum += A_io->on_proc->vals[j];
        }
        
        ASSERT_EQ(A_sten->off_proc->idx1[i+1], A_io->off_proc->idx1[i+1]);
        start = A_sten->off_proc->idx1[i];
        end = A_sten->off_proc->idx1[i+1];
        nnz += end - start;
        for (int j = start; j < end; j++)
        {
            ASSERT_EQ(A_sten->off_proc->idx2[j], A_io->off_proc->idx2[j]);
            ASSERT_NEAR(A_sten->off_proc->vals[j], A_io->off_proc->vals[j], 1e-05);
            sum += A_io->off_proc->vals[j];
        }

        ASSERT_EQ(nnz, row_nnz);
        
        if (fabs(row_sum) > zero_tol)
        {
            ASSERT_NEAR(row_sum, sum, 1e-05);
        }
    }


    fclose(f);

    delete A_io;
    delete A_sten;
    delete[] stencil;
} // end of TEST(ParAnisoTest, TestsInGallery) //

