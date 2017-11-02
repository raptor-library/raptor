// EXPECT_EQ and ASSERT_EQ are macros
// EXPECT_EQ test execution and continues even if there is a failure
// ASSERT_EQ test execution and aborts if there is a failure
// The ASSERT_* variants abort the program execution if an assertion fails 
// while EXPECT_* variants continue with the run.


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
    return RUN_ALL_TESTS();
    MPI_Finalize();
} // end of main() //

TEST(ParCommTest, TestsInCore)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double eps = 0.001;
    double theta = M_PI / 8.0;
    int grid[2] = {10, 10};
    double* stencil = diffusion_stencil_2d(eps, theta);

    CSRMatrix* A_seq = stencil_grid(stencil, grid, 2);

    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);

    ParVector x(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    Vector& x_lcl = x.local;
    for (int i = 0; i < A->local_num_rows; i++)
    {
        x_lcl[i] = A->local_row_map[i];
    }

    A->comm->communicate(x);

    ASSERT_GT(A->off_proc_num_cols, 0);
    for (int i = 0; i < A->off_proc_num_cols; i++)
    {
        ASSERT_NEAR(A->comm->recv_data->buffer[i], A->off_proc_column_map[i], zero_tol);
    }

    double A_dense[10000] = {0};
    for (int i = 0; i < A_seq->n_rows; i++)
    {
        for (int j = A_seq->idx1[i]; j < A_seq->idx1[i+1]; j++)
        {
            A_dense[i*100 + A_seq->idx2[j]] = A_seq->vals[j];
        }
    }

    CSRMatrix* recv_mat = A->comm->communicate(A);
    for (int i = 0; i < A->off_proc_num_cols; i++)
    {
        int global_row = A->off_proc_column_map[i];
        int row_start = recv_mat->idx1[i];
        int row_end = recv_mat->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            int global_col = recv_mat->idx2[j];
            double val = recv_mat->vals[j];
            ASSERT_NEAR(A_dense[global_row*100 + global_col], val, zero_tol);
        }
    }

    delete A;
    delete recv_mat;

} // end of TEST(ParCommTest, TestsInCore) //
