#include <assert.h>
#include <math.h>
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/stencil.hpp"
#include "gallery/par_stencil.hpp"
#include "gallery/diffusion.hpp"

using namespace raptor;

void compare(CSRMatrix& A_seq, ParCSRMatrix& A_par)
{
    std::vector<double> A_dense(A_seq.n_rows * A_seq.n_rows, 0);
    for (int i = 0; i < A_seq.n_rows; i++)
    {
        for (int j = A_seq.idx1[i]; j < A_seq.idx1[i+1]; j++)
        {
            A_dense[i*100 + A_seq.idx2[j]] = A_seq.vals[j];
        }
    }

    int global_nnz;
    MPI_Allreduce(&A_par.local_nnz, &global_nnz, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    assert(A_seq.nnz == global_nnz);

    for (int i = 0; i < A_par.local_num_rows; i++)
    {
        int global_row = A_par.first_local_row + i;
        int row_start = A_par.on_proc->idx1[i];
        int row_end = A_par.on_proc->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            int global_col = A_par.first_local_col + A_par.on_proc->idx2[j];
            double val = A_par.on_proc->vals[j];
            assert(fabs(A_dense[global_row*A_seq.n_rows + global_col] - val)
                    < 1e-05);
        }

        row_start = A_par.off_proc->idx1[i];
        row_end = A_par.off_proc->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            int global_col = A_par.off_proc_column_map[A_par.off_proc->idx2[j]];
            double val = A_par.off_proc->vals[j];
            assert(fabs(A_dense[global_row*A_seq.n_rows + global_col] - val)
                    < 1e-05);
        }
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double eps = 0.001;
    double theta = M_PI/8.0;
    int grid[2] = {10, 10};
    int dense_n = 100;
    double* stencil = diffusion_stencil_2d(eps, theta);

    // Create Sequential Matrix A on each process
    CSRMatrix A;
    stencil_grid(&A, stencil, grid, 2);
    CSRMatrix B;
    stencil_grid(&B, stencil, grid, 2);
    CSRMatrix C;
    A.mult(B, &C);

    // Create Parallel Matrix A_par (and vectors x_par
    // and b_par) and mult b_par <- A_par*x_par
    ParCSRMatrix A_par;
    par_stencil_grid(&A_par, stencil, grid, 2);
    ParCSRMatrix B_par;
    par_stencil_grid(&B_par, stencil, grid, 2);
    ParCSRMatrix C_par;
    A_par.mult(B_par, &C_par);

    // Compare sequential result with parallel result 
    compare(C, C_par);

    // Test ParCSC <- ParCSC*ParCSC
    ParCSCMatrix A_par_csc;
    ParCSCMatrix B_par_csc;
    ParCSCMatrix C_par_csc;
    A_par_csc.copy(&A_par);
    B_par_csc.copy(&B_par);
    A_par_csc.mult(B_par_csc, &C_par_csc);

    // Convert to CSR to compare
    C_par.copy(&C_par_csc);
    compare(C, C_par);

    A_par.mult(B_par_csc, &C_par_csc);
    C_par.copy(&C_par_csc);
    compare(C, C_par);

    A_par.mult(B_par_csc, &C_par);
    compare(C, C_par);

    delete[] stencil;

    MPI_Finalize();
}

