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
    CSRMatrix* A = stencil_grid(stencil, grid, 2);
    A->sort();
    ParCSRMatrix* A_par = par_stencil_grid(stencil, grid, 2);
    A_par->on_proc->sort();
    A_par->off_proc->sort();

    CSRMatrix* S = A->strength(0.0);
    ParCSRMatrix* S_par = A_par->strength(0.0);

    std::vector<double> row_vals(S->n_cols, 0);
    std::vector<int> row_next(S->n_cols);
    int row_start, row_end;
    int global_col, global_row;

    for (int i = 0; i < S_par->local_num_rows; i++)
    {
        global_row = S_par->first_local_row + i;
        row_start = S->idx1[global_row];
        row_end = S->idx1[global_row+1];
        int head = -2;
        int length = 0;
        for (int j = row_start; j < row_end; j++)
        {
            global_col = S->idx2[j];
            row_vals[global_col] = S->vals[j];
            row_next[global_col] = head;
            head = global_col;
            length++;
        }

        row_start = S_par->on_proc->idx1[i];
        row_end = S_par->on_proc->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            global_col = S_par->on_proc->idx2[j] + S_par->first_local_col;
            assert(S_par->on_proc->vals[j] == row_vals[global_col]);
        }

        row_start = S_par->off_proc->idx1[i];
        row_end = S_par->off_proc->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            global_col = S_par->off_proc_column_map[S_par->off_proc->idx2[j]];
            assert(S_par->off_proc->vals[j] == row_vals[global_col]);
        }

        for (int i = 0; i < length; i++)
        {
            row_vals[head] = 0.0;
            head = row_next[head];
        }
    }

    delete S;
    delete A;
    delete S_par;
    delete A_par;

    delete[] stencil;

    MPI_Finalize();
    return 0;
}
