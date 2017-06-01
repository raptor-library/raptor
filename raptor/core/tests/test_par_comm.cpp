#include <assert.h>

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/par_matrix.hpp"
#include "gallery/stencil.hpp"
#include "gallery/par_stencil.hpp"
#include "gallery/diffusion.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double eps = 0.001;
    double theta = M_PI / 8.0;
    int grid[2] = {10, 10};
    double* stencil = diffusion_stencil_2d(eps, theta);

    CSRMatrix A_seq;
    stencil_grid(&A_seq, stencil, grid, 2);

    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);
    ParCSCMatrix* Acsc = new ParCSCMatrix(A);

    ParVector x(A->global_num_rows, A->local_num_rows, A->first_local_row);
    Vector& x_lcl = x.local;
    for (int i = 0; i < A->local_num_rows; i++)
    {
        x_lcl[i] = A->first_local_row + i;
    }

    x.communicate(A->comm);

    assert(A->off_proc_num_cols > 0);
    for (int i = 0; i < A->off_proc_num_cols; i++)
    {
        assert(fabs(A->comm->recv_data->buffer[i] - A->off_proc_column_map[i]) < zero_tol);
    }

    double A_dense[10000] = {0};
    for (int i = 0; i < A_seq.n_rows; i++)
    {
        for (int j = A_seq.idx1[i]; j < A_seq.idx1[i+1]; j++)
        {
            A_dense[i*100 + A_seq.idx2[j]] = A_seq.vals[j];
        }
    }

    CSRMatrix* recv_mat = A->communicate(A->comm);
    for (int i = 0; i < A->off_proc_num_cols; i++)
    {
        int global_row = A->off_proc_column_map[i];
        int row_start = recv_mat->idx1[i];
        int row_end = recv_mat->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            int global_col = recv_mat->idx2[j];
            double val = recv_mat->vals[j];
            assert(fabs(A_dense[global_row*100 + global_col] - val) < zero_tol);
        }
    }

    CSCMatrix* recv_mat_csc = Acsc->communicate(Acsc->comm);
    for (int i = 0; i < recv_mat_csc->n_cols; i++)
    {
        int global_col = recv_mat_csc->col_list[i];
        int col_start = recv_mat_csc->idx1[i];
        int col_end = recv_mat_csc->idx1[i+1];
        for (int j = col_start; j < col_end; j++)
        {
            int row = recv_mat_csc->idx2[j];
            int global_row = A->off_proc_column_map[row];
            double val = recv_mat_csc->vals[j];
            assert(fabs(A_dense[global_row*100 + global_col] - val) < zero_tol);
        }
    }

    delete A;
    delete Acsc;
    delete recv_mat;

    MPI_Finalize();
}



