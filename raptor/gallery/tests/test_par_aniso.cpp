// Copyright (c) 2015-2017, RAPtor Developer Team, University of Illinois at Urbana-Champaign
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

    const char* A0_fn = "../../../../test_data/aniso.pm";

    int row;
    int start, end;
    int grid[2] = {25, 25};
    double eps = 0.001;
    double theta = M_PI/8.0;
    double* stencil = diffusion_stencil_2d(eps, theta);
    ParCSRMatrix* A_sten = par_stencil_grid(stencil, grid, 2);
    ParCSRMatrix* A_io = readParMatrix(A0_fn);

    // Compare shapes
    ASSERT_EQ(A_io->global_num_rows, A_sten->global_num_rows);
    ASSERT_EQ(A_io->global_num_cols, A_sten->global_num_cols);

    ASSERT_EQ(A_sten->local_num_rows, A_io->local_num_rows);
    ASSERT_EQ(A_sten->on_proc_num_cols, A_io->on_proc_num_cols);
    ASSERT_EQ(A_sten->partition->first_local_row, A_io->partition->first_local_row);
    ASSERT_EQ(A_sten->partition->last_local_row, A_io->partition->last_local_row);
    ASSERT_EQ(A_sten->partition->first_local_col, A_io->partition->first_local_col);
    ASSERT_EQ(A_sten->partition->last_local_col, A_io->partition->last_local_col);

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

    for (int i = 0; i < A_sten->local_num_rows; i++)
    {
        ASSERT_EQ(A_sten->on_proc->idx1[i+1], A_io->on_proc->idx1[i+1]);
        start = A_sten->on_proc->idx1[i];
        end = A_sten->on_proc->idx1[i+1];

        for (int j = start; j < end; j++)
        {
            ASSERT_EQ(A_sten->on_proc->idx2[j], A_io->on_proc->idx2[j]);
            ASSERT_NEAR(A_sten->on_proc->vals[j], A_io->on_proc->vals[j], 1e-05);
        }
        
        ASSERT_EQ(A_sten->off_proc->idx1[i+1], A_io->off_proc->idx1[i+1]);
        start = A_sten->off_proc->idx1[i];
        end = A_sten->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            ASSERT_EQ(A_sten->off_proc->idx2[j], A_io->off_proc->idx2[j]);
            ASSERT_NEAR(A_sten->off_proc->vals[j], A_io->off_proc->vals[j], 1e-05);
        }
    }

    delete A_io;
    delete A_sten;
    delete[] stencil;
} // end of TEST(ParAnisoTest, TestsInGallery) //

