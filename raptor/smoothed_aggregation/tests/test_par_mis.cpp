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
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);
    A->on_proc->sort();
    A->off_proc->sort();
    ParCSRMatrix* S = A->strength(0.0);

    ParCSRMatrix* S_sq = S->mult(S);
    S_sq->comm = new ParComm(S_sq->off_proc_column_map, S_sq->first_local_row,
            S_sq->first_local_col, S_sq->global_num_cols, S_sq->local_num_cols);

    std::vector<int> local_states;
    std::vector<int> off_proc_states;
    int local_coarse = S_sq->maximal_independent_set(local_states, off_proc_states);
    for (std::vector<int>::iterator it = local_states.begin(); it != local_states.end(); ++it)
    {
        if (*it != 0 && *it != 1)
            printf("LocalState = %d\n", *it);
    }

    delete S_sq;
    delete S;
    delete A;

    delete[] stencil;

    MPI_Finalize();
    return 0;
}

